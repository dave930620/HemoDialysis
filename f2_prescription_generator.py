#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2 prescription generator (print-focused)
- Patient-wise split (as requested to avoid leakage): 0.8 / 0.1 / 0.1 by PatientID
- Saves ONLY plots to disk; all metrics are printed to terminal clearly
- Aligns exactly to f1's feature_info.pkl (column order, scaling, imputation)
- Loads f1 SAINT directly and infers hidden_size from the checkpoint (silent init)
- Two-stage training: Stage A (doc mimic), Stage B (effect + doc-prox + proximal teacher)
- Hard output constraints at inference (int / 0.5 / 0.25 steps)
- Uses matplotlib (Agg); ASCII-only filenames; no seaborn
- Fixes: sklearn feature-name warning; pandas dtype FutureWarning; tensor-wrapping warning
"""

import os, re, math, pickle, random, io, contextlib
from pathlib import Path
from typing import List, Tuple
import re

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- paths & constants --------------------
DATA_CSV = os.environ.get("DATA_CSV", "cleaned_data_after_fill_and_drop.csv")
FEATURE_INFO_PATH = os.environ.get("FEATURE_INFO", "feature_info.pkl")
F1_MODEL_PATH = os.environ.get("F1_MODEL", "saint_pd_model.pth")
REPORT_DIR = os.environ.get("REPORT_DIR", "report_model2")

PATIENT_ID_COL = "PatientID"
OUTCOME_COL = "PD Kt/V"

CAT_RX = "long term PD system"
CONT_RX = ["No. of bag/day", "total vol/day", "glucose_total", "calcium_total"]
STEP_MAP = {"No. of bag/day": 1.0, "total vol/day": 0.5, "glucose_total": 0.25, "calcium_total": 0.25}

SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.75, 0.1, 0.15  # patient-wise
BATCH_SIZE = 256
NUM_WORKERS = 2
USE_AMP = True
GRAD_CLIP = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Stage A
LR_A = 1e-3
EPOCHS_A = 5
HIDDEN = 128
DROPOUT = 0.25

# Stage B
LR_B = 3e-4
EPOCHS_B = 25

# Trust-region / proximal / evaluation windows
# NOTE: use *separate* trust regions for PASS vs FAIL to match the goal:
#   - FAIL (doctor predicted outcome < THR): explore more  -> larger eps
#   - PASS (doctor predicted outcome >= THR): stay close   -> smaller eps
EPS_CONT_Z_PASS = 0.75
EPS_CONT_Z_FAIL = 3.00
EPS_CAT_SOFT_PASS = 0.15
EPS_CAT_SOFT_FAIL = 0.35

LAMBDA_CONSTRAINT = 1.0
LAMBDA_PROX_CONT = 0.05
LAMBDA_PROX_CAT = 0.01
DELTA_WIN = 0

# Extra objective: in FAIL cases, explicitly push predicted Kt/V above threshold.
# This targets your evaluation metric (P_pass in Group 1).
LAMBDA_THR_FAIL = 2.0
LAMBDA_THR_PASS = 0.0

# Extra threshold-push term (only meaningful for FAIL cases):
# encourages crossing the clinical threshold rather than just small improvements.
LAMBDA_THR_FAIL = 1.0
LAMBDA_THR_PASS = 0.0

# -------------------- conditional (PASS/FAIL) gating for Stage B --------------------
THR_GATE = 1.7  # doctor outcome threshold
# When doctor outcome >= THR_GATE: keep prescriptions close; only mild effect push
W_EFFECT_PASS = 0.05
# When doctor outcome < THR_GATE: prioritize improving predicted outcome
W_EFFECT_FAIL = 5.00
# In failing cases, relax proximal-to-teacher pressure
W_PROX_FAIL = 0.00

CURR_STRONG_FRAC = 0.3
CURR_MID_FRAC = 0.7

# -------------------- helpers --------------------
SAFE_CHARS = re.compile(r"[^A-Za-z0-9_\- ]+")
def sanitize(s: str) -> str:
    return SAFE_CHARS.sub("_", str(s))

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def safe_name(s: str) -> str:
    """
    Make a string safe for filenames.
    """
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_\- ]+", "", s)
    return s.strip().replace(" ", "_")


def plot_loss_curves(train_losses, val_losses, prefix: str):
    """Plot train/val loss curves (PNG only, no CSV)."""
    ensure_dir(REPORT_DIR)
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix} loss curve")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, f"{prefix}_loss.png")
    plt.savefig(out_path)
    plt.close()
    print(f"{prefix} loss curve saved to: {out_path}")

# -------------------- import f1 + feature_info --------------------
import importlib.util as _imp
def import_f1(path="f1.py"):
    spec = _imp.spec_from_file_location("f1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import f1.py")
    m = _imp.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

f1 = import_f1("f1.py")

with open(FEATURE_INFO_PATH, "rb") as f:
    FEATURE_INFO = pickle.load(f)

ORIG_FEATURES: List[str] = FEATURE_INFO["original_feature_names"]
DISCRETE_IDX: List[int] = FEATURE_INFO["discrete_feature_indices"]
CONTINUOUS_IDX: List[int] = FEATURE_INFO["continuous_feature_indices"]
DISCRETE_NAMES = [ORIG_FEATURES[i] for i in DISCRETE_IDX]
CONTINUOUS_NAMES = [ORIG_FEATURES[i] for i in CONTINUOUS_IDX]
IMPUTER = FEATURE_INFO["imputer"]
SCALER = FEATURE_INFO["scaler"]

for col in [CAT_RX] + CONT_RX:
    if col not in ORIG_FEATURES:
        raise ValueError(f"Required Rx column '{col}' is not in feature_info['original_feature_names'].")

# -------------------- load f1 SAINT cleanly --------------------
def build_f1_model_from_ckpt(state_dict: dict, input_size: int, disc_idx: List[int], cont_idx: List[int]) -> nn.Module:
    d_model = None
    for k, v in state_dict.items():
        if k.startswith("discrete_embedding.0.weight") or k.startswith("continuous_embedding.0.weight"):
            d_model = v.shape[0]
            break
    if d_model is None:
        for k, v in state_dict.items():
            if "self_attn.in_proj_weight" in k:
                d_model = v.shape[1]
                break
    if d_model is None:
        raise RuntimeError("Cannot infer F1 hidden_size from checkpoint.")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = f1.SAINT(
            input_size=input_size,
            hidden_size=d_model,
            output_size=1,
            discrete_feature_indices=disc_idx,
            continuous_feature_indices=cont_idx
        )
    model.to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def load_f1_model(model_path: str) -> nn.Module:
    sd = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model = build_f1_model_from_ckpt(sd, input_size=len(ORIG_FEATURES),
                                     disc_idx=DISCRETE_IDX, cont_idx=CONTINUOUS_IDX)
    return model

F1_MODEL = load_f1_model(F1_MODEL_PATH)
# Freeze F1 parameters (we still allow gradients to flow through its computation graph to F2 outputs)
for _p in F1_MODEL.parameters():
    _p.requires_grad_(False)

# -------------------- preprocessing aligned to f1 --------------------
def standardize_like_f1(df_part: pd.DataFrame) -> pd.DataFrame:
    for c in ORIG_FEATURES:
        if c not in df_part.columns:
            raise ValueError(f"Missing feature '{c}' for F1 standardization.")

    X = df_part[ORIG_FEATURES].copy()

    if len(CONTINUOUS_NAMES) > 0:
        cont_df = X[CONTINUOUS_NAMES].copy().astype("float64")
        cont_imp = IMPUTER.transform(cont_df)     # keeps names; no warning
        cont_std = SCALER.transform(cont_imp)

        X[CONTINUOUS_NAMES] = X[CONTINUOUS_NAMES].astype("float64")
        X.loc[:, CONTINUOUS_NAMES] = cont_std

    return X

@torch.no_grad()
def f1_predict_ktv_from_raw(df_raw: pd.DataFrame) -> np.ndarray:
    Xdf = standardize_like_f1(df_raw)
    X = torch.tensor(Xdf.values, dtype=torch.float32, device=DEVICE)
    pred = F1_MODEL(X).squeeze(-1).detach().cpu().numpy()
    return pred

# -------------------- data split & dataset --------------------
def patient_split(df: pd.DataFrame, pid_col: str, tr=0.8, va=0.1, te=0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pids = df[pid_col].astype(str).unique()
    def sh(x): return (hash(x) % 10**9) / 10**9
    bins = {pid: sh(pid) for pid in pids}
    def bucket(pid):
        v = bins[str(pid)]
        return "train" if v < tr else ("val" if v < tr+va else "test")
    part = df[pid_col].map(bucket)
    return df[part=="train"].copy(), df[part=="val"].copy(), df[part=="test"].copy()

class RxDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        # Each row is one *record*; split is *patient-wise* to avoid leakage
        self.df = df.reset_index(drop=True)

        Xstd = standardize_like_f1(self.df)

        for c in [CAT_RX] + CONT_RX:
            if c in Xstd.columns:
                if c == CAT_RX:
                    Xstd[c] = 0
                else:
                    Xstd[c] = Xstd[c].astype("float64")
                    Xstd[c] = 0.0

        self.X = Xstd.values.astype(np.float32)

        self.y_cont = np.zeros((len(df), len(CONT_RX)), dtype=np.float32)
        for j, c in enumerate(CONT_RX):
            mu = SCALER.mean_[CONTINUOUS_NAMES.index(c)]
            sd = SCALER.scale_[CONTINUOUS_NAMES.index(c)] or 1.0
            z = (df[c].values.astype(np.float32) - mu) / sd
            self.y_cont[:, j] = z
        self.y_cat = df[CAT_RX].astype(int).values
        # True measured outcome (used only for analysis/eval prints)
        self.y_outcome = df[OUTCOME_COL].astype(float).values.astype(np.float32)

        # IMPORTANT: gate PASS/FAIL using the *predictable* signal available at inference:
        # F1's predicted Kt/V if we keep the doctor's prescription (i.e., current RX).
        # This avoids train-time leakage from using true outcome as a gate.
        try:
            self.y_doc_hat = f1_predict_ktv_from_raw(df[ORIG_FEATURES]).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed computing F1 doctor-hat predictions for gating: {e}")

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        return self.X[idx], self.y_cont[idx], self.y_cat[idx], self.y_outcome[idx], self.y_doc_hat[idx]

# -------------------- small F2 model --------------------
class F2RxHead(nn.Module):
    def __init__(self, in_dim: int, n_cont: int, n_cat: int, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.head_cont = nn.Linear(hidden, n_cont)     # z-space
        self.head_cat = nn.Linear(hidden, n_cat)       # logits

    def forward(self, x):
        h = self.backbone(x)
        return self.head_cont(h), self.head_cat(h)

# -------------------- loss helpers --------------------
def hinge_penalty(dist: torch.Tensor, eps) -> torch.Tensor:
    """Hinge penalty max(0, dist - eps).

    Args:
        dist: (B,) distances
        eps:  scalar float or (B,) tensor
    """
    if not torch.is_tensor(eps):
        eps = torch.tensor(float(eps), device=dist.device, dtype=dist.dtype)
    return torch.relu(dist - eps).mean()

def kl_div(p_now: torch.Tensor, p_ref: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return (p_now * (torch.log(p_now + eps) - torch.log(p_ref + eps))).sum(dim=1).mean()

def doc_distance(z_pred: torch.Tensor, z_doc: torch.Tensor, p_now: torch.Tensor, y_doc_cat: torch.Tensor,
                 a_cont=1.0, a_cat=1.0) -> torch.Tensor:
    d_cont = torch.linalg.vector_norm(z_pred - z_doc, dim=1)  # L2 in z
    p_doc = torch.gather(p_now, 1, y_doc_cat.view(-1, 1)).squeeze(1)
    d_cat = 1.0 - p_doc
    return a_cont * d_cont + a_cat * d_cat

# -------------------- device transfer --------------------
def to_device(batch):
    # Support:
    #  - (xb, yb_cont, yb_cat, yb_outcome_true, yb_doc_hat)
    #  - (xb, yb_cont, yb_cat, yb_outcome_true)
    #  - (xb, yb_cont, yb_cat)
    if len(batch) == 5:
        xb, yb_cont, yb_cat, yb_out, yb_doc_hat = batch
    elif len(batch) == 4:
        xb, yb_cont, yb_cat, yb_out = batch
        yb_doc_hat = None
    else:
        xb, yb_cont, yb_cat = batch
        yb_out = None
        yb_doc_hat = None

    if torch.is_tensor(xb):
        xb = xb.detach().to(dtype=torch.float32, device=DEVICE, non_blocking=True)
    else:
        xb = torch.as_tensor(xb, dtype=torch.float32, device=DEVICE)

    if torch.is_tensor(yb_cont):
        yb_cont = yb_cont.detach().to(dtype=torch.float32, device=DEVICE, non_blocking=True)
    else:
        yb_cont = torch.as_tensor(yb_cont, dtype=torch.float32, device=DEVICE)

    if torch.is_tensor(yb_cat):
        yb_cat = yb_cat.detach().to(dtype=torch.long, device=DEVICE, non_blocking=True)
    else:
        yb_cat = torch.as_tensor(yb_cat, dtype=torch.long, device=DEVICE)

    if yb_out is not None:
        if torch.is_tensor(yb_out):
            yb_out = yb_out.detach().to(dtype=torch.float32, device=DEVICE, non_blocking=True)
        else:
            yb_out = torch.as_tensor(yb_out, dtype=torch.float32, device=DEVICE)

    if yb_doc_hat is not None:
        if torch.is_tensor(yb_doc_hat):
            yb_doc_hat = yb_doc_hat.detach().to(dtype=torch.float32, device=DEVICE, non_blocking=True)
        else:
            yb_doc_hat = torch.as_tensor(yb_doc_hat, dtype=torch.float32, device=DEVICE)

    return xb, yb_cont, yb_cat, yb_out, yb_doc_hat

# -------------------- Stage A --------------------
def train_stage_A(model, dl_tr, dl_va):
    model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR_A, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP and DEVICE=='cuda')
    ce = nn.CrossEntropyLoss()
    best_val = float('inf')

    train_losses = []
    val_losses = []

    print("\n=== Stage A (doc mimic) ===")
    for epoch in range(1, EPOCHS_A+1):
        model.train(); tr_sum=0; ntr=0
        for batch in dl_tr:
            xb, yb_cont, yb_cat, _, _ = to_device(batch)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP and DEVICE=='cuda'):
                z_pred, cat_logits = model(xb)
                loss = ((z_pred - yb_cont)**2).mean() + ce(cat_logits, yb_cat)
            scaler.scale(loss).backward()
            if GRAD_CLIP:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
            bs = xb.size(0); tr_sum += loss.item()*bs; ntr += bs

        model.eval(); va_sum=0; nva=0
        with torch.no_grad():
            for batch in dl_va:
                xb, yb_cont, yb_cat, _, _ = to_device(batch)
                z_pred, cat_logits = model(xb)
                loss = ((z_pred - yb_cont)**2).mean() + ce(cat_logits, yb_cat)
                bs = xb.size(0); va_sum += loss.item()*bs; nva += bs

        tr_loss = tr_sum/max(1,ntr); va_loss = va_sum/max(1,nva)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        print(f"[A] epoch {epoch:03d}  train_loss={tr_loss:.6f}  val_loss={va_loss:.6f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), os.path.join(REPORT_DIR, "f2_stageA_best.pth"))

    # draw loss curves (PNG only)
    plot_loss_curves(train_losses, val_losses, prefix="stageA")

# -------------------- F1 bridge for Stage B --------------------
def predict_f1_ktv_from_z_batch(x_std_batch: torch.Tensor, z_cont_pred: torch.Tensor, p_cat: torch.Tensor) -> torch.Tensor:
    """Differentiable F1 prediction from a standardized feature batch.

    Notes
    -----
    - x_std_batch is already standardized in the *exact* space F1 expects (ORIG_FEATURES order).
    - We write the generated prescription back into the standardized feature vector, then call F1_MODEL directly.
    - Continuous RX is fully differentiable.
    - Categorical RX is inserted via argmax index (non-differentiable through the discrete choice),
      because F1 consumes categorical features as indices. If you want cat gradients, you'd need
      an F1 variant that accepts a soft/expected embedding.
    """
    Xstd = x_std_batch.clone().float()

    # write continuous prescription (z-space)
    for j, c in enumerate(CONT_RX):
        col_idx = ORIG_FEATURES.index(c)
        Xstd[:, col_idx] = z_cont_pred[:, j]

    # write categorical prescription index (still float in the input tensor; F1 will cast internally if needed)
    cat_idx = torch.argmax(p_cat, dim=1).to(dtype=torch.float32)
    Xstd[:, ORIG_FEATURES.index(CAT_RX)] = cat_idx

    # IMPORTANT: do NOT detach; we want gradients to flow into z_cont_pred through F1
    ktv = F1_MODEL(Xstd).squeeze(-1)
    return ktv

# -------------------- Stage B --------------------
def train_stage_B(model, dl_tr, dl_va, teacher_state_path: str, num_classes: int):
    teacher = F2RxHead(len(ORIG_FEATURES), len(CONT_RX), n_cat=num_classes).to(DEVICE)
    teacher.load_state_dict(torch.load(teacher_state_path, map_location=DEVICE, weights_only=True))
    teacher.eval()

    opt = optim.AdamW(model.parameters(), lr=LR_B, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP and DEVICE=='cuda')
    total_steps = EPOCHS_B * max(1, len(dl_tr)); step=0

    train_losses = []
    val_losses = []

    print("\n=== Stage B (effect + doc-prox + proximal teacher) ===")
    for epoch in range(1, EPOCHS_B+1):
        model.train(); tr_sum=0; ntr=0
        for batch in dl_tr:
            xb, yb_cont, yb_cat, yb_out, yb_doc_hat = to_device(batch)

            with torch.no_grad():
                z_bc, logit_bc = teacher(xb)
                p_bc = torch.softmax(logit_bc, dim=1)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP and DEVICE=='cuda'):
                z_pred, logit = model(xb)
                p_now = torch.softmax(logit, dim=1)

                ktv_t = predict_f1_ktv_from_z_batch(xb, z_pred, p_now)

                # Conditional gating by *doctor-hat* outcome (PASS vs FAIL)
                # (available at inference; avoids leakage)
                if yb_doc_hat is None:
                    gate = torch.ones_like(ktv_t)
                else:
                    gate = (yb_doc_hat >= THR_GATE).to(dtype=torch.float32)
                w_effect = gate * W_EFFECT_PASS + (1.0 - gate) * W_EFFECT_FAIL
                loss_effect = - (w_effect * ktv_t).mean()

                # In FAIL cases, explicitly push Kt/V above threshold.
                # softplus(thr - y) behaves like a smooth hinge.
                thr_push = torch.nn.functional.softplus(THR_GATE - ktv_t)
                w_thr = gate * LAMBDA_THR_PASS + (1.0 - gate) * LAMBDA_THR_FAIL
                loss_thr = (w_thr * thr_push).mean()

                d = doc_distance(z_pred, yb_cont, p_now, yb_cat, 1.0, 1.0)
                eps = gate * (EPS_CONT_Z_PASS + EPS_CAT_SOFT_PASS) + (1.0 - gate) * (EPS_CONT_Z_FAIL + EPS_CAT_SOFT_FAIL)
                loss_constraint = LAMBDA_CONSTRAINT * hinge_penalty(d, eps)

                # Proximal to teacher, but relax for FAIL cases
                cont_prox_per = ((z_pred - z_bc)**2).mean(dim=1)
                # KL per sample
                kl_per = (p_now * (torch.log(p_now + 1e-8) - torch.log(p_bc + 1e-8))).sum(dim=1)
                if yb_doc_hat is None:
                    w_prox = torch.ones_like(cont_prox_per)
                else:
                    gate = (yb_doc_hat >= THR_GATE).to(dtype=torch.float32)
                    w_prox = gate * 1.0 + (1.0 - gate) * W_PROX_FAIL
                loss_prox = (w_prox * (LAMBDA_PROX_CONT * cont_prox_per + LAMBDA_PROX_CAT * kl_per)).mean()

                frac = step / max(1, total_steps)
                lc = 1.2 if frac < CURR_STRONG_FRAC else (1.0 if frac < CURR_MID_FRAC else 0.9)
                lp = lc

                loss = loss_effect + loss_thr + lc*loss_constraint + lp*loss_prox

            scaler.scale(loss).backward()
            if GRAD_CLIP:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()

            bs = xb.size(0); tr_sum += loss.item()*bs; ntr += bs; step += 1

        model.eval(); va_sum=0; nva=0
        with torch.no_grad():
            for batch in dl_va:
                xb, yb_cont, yb_cat, yb_out, yb_doc_hat = to_device(batch)
                z_pred, logit = model(xb)
                p_now = torch.softmax(logit, dim=1)
                ktv_t = predict_f1_ktv_from_z_batch(xb, z_pred, p_now)

                # Conditional gating by doctor-hat outcome (PASS vs FAIL)
                if yb_doc_hat is None:
                    gate = torch.ones_like(ktv_t)
                else:
                    gate = (yb_doc_hat >= THR_GATE).to(dtype=torch.float32)
                w_effect = gate * W_EFFECT_PASS + (1.0 - gate) * W_EFFECT_FAIL
                loss_effect = - (w_effect * ktv_t).mean()

                thr_push = torch.nn.functional.softplus(THR_GATE - ktv_t)
                w_thr = gate * LAMBDA_THR_PASS + (1.0 - gate) * LAMBDA_THR_FAIL
                loss_thr = (w_thr * thr_push).mean()
                d = doc_distance(z_pred, yb_cont, p_now, yb_cat, 1.0, 1.0)
                eps = gate * (EPS_CONT_Z_PASS + EPS_CAT_SOFT_PASS) + (1.0 - gate) * (EPS_CONT_Z_FAIL + EPS_CAT_SOFT_FAIL)
                loss_constraint = LAMBDA_CONSTRAINT * hinge_penalty(d, eps)
                z_bc, logit_bc = teacher(xb); p_bc = torch.softmax(logit_bc, dim=1)
                # Proximal to teacher, but relax for FAIL cases
                cont_prox_per = ((z_pred - z_bc)**2).mean(dim=1)
                # KL per sample
                kl_per = (p_now * (torch.log(p_now + 1e-8) - torch.log(p_bc + 1e-8))).sum(dim=1)
                if yb_doc_hat is None:
                    w_prox = torch.ones_like(cont_prox_per)
                else:
                    gate = (yb_doc_hat >= THR_GATE).to(dtype=torch.float32)
                    w_prox = gate * 1.0 + (1.0 - gate) * W_PROX_FAIL
                loss_prox = (w_prox * (LAMBDA_PROX_CONT * cont_prox_per + LAMBDA_PROX_CAT * kl_per)).mean()
                loss_val = loss_effect + loss_thr + loss_constraint + loss_prox
                bs = xb.size(0); va_sum += loss_val.item()*bs; nva += bs

        tr_loss = tr_sum/max(1,ntr); va_loss = va_sum/max(1,nva)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        print(f"[B] epoch {epoch:03d}  train_loss={tr_loss:.6f}  val_loss={va_loss:.6f}")

    # draw loss curves (PNG only)
    plot_loss_curves(train_losses, val_losses, prefix="stageB")

# -------------------- evaluation --------------------
def evaluate_and_plots(df_eval: pd.DataFrame, model_rx_df: pd.DataFrame, prefix: str):
    ensure_dir(REPORT_DIR)

    print(f"\n=== Evaluation ({prefix}) ===")

    # ----- PASS/FAIL split (based on doctor's outcome) -----
    if OUTCOME_COL in df_eval.columns:
        _doc_out = df_eval[OUTCOME_COL].to_numpy().astype(float)
        mask_pass = _doc_out >= THR_GATE
        mask_fail = ~mask_pass
    else:
        mask_pass = None
        mask_fail = None

    def _cont_metrics(gt_arr: np.ndarray, pr_arr: np.ndarray):
        mae = float(np.mean(np.abs(pr_arr - gt_arr))) if gt_arr.size else float("nan")
        rmse = float(np.sqrt(np.mean((pr_arr - gt_arr) ** 2))) if gt_arr.size else float("nan")
        if gt_arr.size and (np.std(gt_arr) > 0 and np.std(pr_arr) > 0):
            r = float(np.corrcoef(gt_arr, pr_arr)[0, 1])
        else:
            r = float("nan")
        return mae, rmse, r

    # ----- continuous metrics -----
    for col in CONT_RX:
        gt = df_eval[col].to_numpy().astype(float)
        pr = model_rx_df[col].to_numpy().astype(float)

        mae, rmse, r = _cont_metrics(gt, pr)
        print(f"[{prefix}][CONT][ALL] {col}")
        print(f"    MAE   (|model - doctor|)      = {mae:.6f}")
        print(f"    RMSE  (sqrt MSE)              = {rmse:.6f}")
        print(f"    Corr. (Pearson correlation)   = {r:.6f}")

        if mask_pass is not None:
            if mask_pass.sum() > 0:
                mae_p, rmse_p, r_p = _cont_metrics(gt[mask_pass], pr[mask_pass])
                print(f"[{prefix}][CONT][PASS] {col}  N={int(mask_pass.sum())}")
                print(f"    MAE   (|model - doctor|)      = {mae_p:.6f}")
                print(f"    RMSE  (sqrt MSE)              = {rmse_p:.6f}")
                print(f"    Corr. (Pearson correlation)   = {r_p:.6f}")
            if mask_fail.sum() > 0:
                mae_f, rmse_f, r_f = _cont_metrics(gt[mask_fail], pr[mask_fail])
                print(f"[{prefix}][CONT][FAIL] {col}  N={int(mask_fail.sum())}")
                print(f"    MAE   (|model - doctor|)      = {mae_f:.6f}")
                print(f"    RMSE  (sqrt MSE)              = {rmse_f:.6f}")
                print(f"    Corr. (Pearson correlation)   = {r_f:.6f}")

        plt.figure(figsize=(6,6))
        plt.scatter(gt, pr, s=10)
        plt.xlabel("Doctor")
        plt.ylabel("Model")
        plt.title(f"{prefix}: {col}")
        plt.grid(True, alpha=0.25)
        out = os.path.join(REPORT_DIR, f"{prefix}_scatter_{safe_name(col)}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"    Scatter plot saved to: {out}")

# ----- categorical metrics -----
    gt_cat = df_eval[CAT_RX].to_numpy().astype(int)
    pr_cat = model_rx_df[CAT_RX].to_numpy().astype(int)
    acc = float((gt_cat==pr_cat).mean()) if len(gt_cat)>0 else float('nan')

    print(f"\n[{prefix}][CAT] {CAT_RX}")
    print(f"    Accuracy (exact match rate) = {acc:.6f}")
    if mask_pass is not None:
        if mask_pass.sum() > 0:
            acc_p = float((gt_cat[mask_pass] == pr_cat[mask_pass]).mean())
            print(f"    [PASS] Accuracy N={int(mask_pass.sum())} = {acc_p:.6f}")
        if mask_fail.sum() > 0:
            acc_f = float((gt_cat[mask_fail] == pr_cat[mask_fail]).mean())
            print(f"    [FAIL] Accuracy N={int(mask_fail.sum())} = {acc_f:.6f}")

    K = int(max(gt_cat.max(), pr_cat.max())+1) if len(gt_cat)>0 else 0
    if K>0:
        cm = np.zeros((K,K), dtype=int)
        for g,p in zip(gt_cat, pr_cat):
            if 0<=g<K and 0<=p<K:
                cm[g,p]+=1
        plt.figure(figsize=(6,5))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"{prefix} Confusion {sanitize(CAT_RX)}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        cm_path = os.path.join(REPORT_DIR, f"{prefix}_cm_{sanitize(CAT_RX)}.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"    Confusion matrix saved to: {cm_path}")
        print("    Confusion matrix (rows=true, cols=pred):")
        for i in range(K):
            row_str = " ".join(f"{cm[i,j]:4d}" for j in range(K))
            print(f"        class {i}: {row_str}")

    # ----- threshold analyses (doctor vs model outcome) -----
    doctor_outcome = df_eval[OUTCOME_COL].to_numpy().astype(float)

    # patch original features with model prescription, then run F1
    patched = df_eval[ORIG_FEATURES].copy()
    for c in [CAT_RX] + CONT_RX:
        patched[c] = model_rx_df[c].values
    ktv_model = f1_predict_ktv_from_raw(patched)

    thr = 1.7
    delta = DELTA_WIN

    # doctors below threshold
    mask_d = doctor_outcome < thr
    Nd = int(mask_d.sum())
    if Nd > 0:
        p_pass = float((ktv_model[mask_d] >= thr).mean())
        p_win = float((ktv_model[mask_d] > doctor_outcome[mask_d] + delta).mean())
        mwin = (ktv_model[mask_d] > doctor_outcome[mask_d] + delta)
        p_pass_given_win = float((ktv_model[mask_d][mwin] >= thr).mean()) if mwin.any() else float('nan')
    else:
        p_pass = p_win = p_pass_given_win = float('nan')

    # doctors already above threshold
    mask_e = doctor_outcome >= thr
    Ne = int(mask_e.sum())
    if Ne > 0:
        p_fail = float((ktv_model[mask_e] < thr).mean())
        p_win_e = float((ktv_model[mask_e] > doctor_outcome[mask_e] + delta).mean())
        mlose = (ktv_model[mask_e] <= doctor_outcome[mask_e] + delta)
        p_fail_given_lose = float((ktv_model[mask_e][mlose] < thr).mean()) if mlose.any() else float('nan')
    else:
        p_fail = p_win_e = p_fail_given_lose = float('nan')

    print(f"\n[{prefix}][THR] Threshold summary (thr = {thr}, delta = {delta})")
    print(f"    Group 1: doctor outcome < {thr}")
    print(f"        Nd (count)                 = {Nd}")
    if Nd > 0:
        print(f"        P_pass        = {p_pass:.6f}  "
              f"(model prescription pushes patient to >= {thr})")
        print(f"        P_win         = {p_win:.6f}  "
              f"(model improves Kt/V by more than delta over doctor)")
        print(f"        P_pass|win    = {p_pass_given_win:.6f}  "
              f"(among 'wins', fraction that also reach >= {thr})")
    else:
        print("        No cases in this group.")

    print(f"\n    Group 2: doctor outcome >= {thr}")
    print(f"        Ne (count)                 = {Ne}")
    if Ne > 0:
        print(f"        P_fail        = {p_fail:.6f}  "
              f"(model prescription falls below {thr})")
        print(f"        P_win         = {p_win_e:.6f}  "
              f"(model still improves > delta vs doctor)")
        print(f"        P_fail|lose   = {p_fail_given_lose:.6f}  "
              f"(among cases where model not clearly better, "
              f"fraction that drop below {thr})")
    else:
        print("        No cases in this group.")

# -------------------- inference with hard constraints --------------------
def infer_with_constraints(model: nn.Module, df_part: pd.DataFrame) -> pd.DataFrame:
    model.eval(); rows=[]

    # Gate PASS/FAIL using F1's prediction for the *doctor* prescription.
    # This is available at inference and matches the training gate.
    doc_hat = f1_predict_ktv_from_raw(df_part[ORIG_FEATURES]).astype(np.float32)
    gate_pass = (doc_hat >= THR_GATE)

    Xstd = standardize_like_f1(df_part)
    for c in [CAT_RX] + CONT_RX:
        if c == CAT_RX:
            Xstd[c] = 0
        else:
            Xstd[c] = Xstd[c].astype("float64")
            Xstd[c] = 0.0

    Xmat = Xstd.values.astype(np.float32)

    N = Xmat.shape[0]; bs = BATCH_SIZE
    for i in range(0, N, bs):
        xb = torch.as_tensor(Xmat[i:i+bs], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            z_pred, logit = model(xb)
            p_now = torch.softmax(logit, dim=1)
            cat_idx = torch.argmax(p_now, dim=1).cpu().numpy().astype(int)

            z_doc = np.zeros((z_pred.size(0), len(CONT_RX)), dtype=np.float32)
            df_chunk = df_part.iloc[i:i+bs]
            for j, c in enumerate(CONT_RX):
                mu = SCALER.mean_[CONTINUOUS_NAMES.index(c)]
                sd = SCALER.scale_[CONTINUOUS_NAMES.index(c)] or 1.0
                raw_vals = df_chunk[c].values.astype(np.float32)
                z_doc[:, j] = (raw_vals - mu)/sd

            z_np = z_pred.cpu().numpy()
            z_proj = np.zeros_like(z_np)
            for r in range(z_np.shape[0]):
                eps_cont = EPS_CONT_Z_PASS if gate_pass[i + r] else EPS_CONT_Z_FAIL
                d = z_np[r] - z_doc[r]
                n = np.linalg.norm(d)
                z_proj[r] = z_np[r] if (n==0 or n <= eps_cont) else (z_doc[r] + d * (eps_cont/n))

            denorm_cols = {}
            for j, c in enumerate(CONT_RX):
                mu = SCALER.mean_[CONTINUOUS_NAMES.index(c)]
                sd = SCALER.scale_[CONTINUOUS_NAMES.index(c)] or 1.0
                raw = z_proj[:, j]*sd + mu
                step = STEP_MAP[c]
                raw = np.round(raw/step)*step
                if c == "No. of bag/day":
                    raw = np.round(raw)
                denorm_cols[c] = raw

            for r in range(z_proj.shape[0]):
                row = {CAT_RX: int(cat_idx[r])}
                for c in CONT_RX:
                    row[c] = float(denorm_cols[c][r])
                rows.append(row)

    return pd.DataFrame(rows, index=df_part.index)

# -------------------- main --------------------
def main():
    ensure_dir(REPORT_DIR)

    df_raw = pd.read_csv(DATA_CSV)

    missing = [c for c in ORIG_FEATURES if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns required by f1: {missing}")
    if OUTCOME_COL not in df_raw.columns:
        raise ValueError(f"Input CSV missing outcome column '{OUTCOME_COL}'")
    if PATIENT_ID_COL not in df_raw.columns:
        raise ValueError(f"Input CSV missing PatientID column '{PATIENT_ID_COL}'")

    df_use = df_raw[[PATIENT_ID_COL, OUTCOME_COL] + ORIG_FEATURES].copy()

    # Patient-wise split (each row is one record; split is grouped by PatientID)
    tr, va, te = patient_split(df_use, PATIENT_ID_COL, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)
    print(f"Split sizes (records): train={len(tr)}, val={len(va)}, test={len(te)}  | grouped by patients")

    ds_tr = RxDataset(tr[ORIG_FEATURES + [OUTCOME_COL, PATIENT_ID_COL]])
    ds_va = RxDataset(va[ORIG_FEATURES + [OUTCOME_COL, PATIENT_ID_COL]])
    ds_te = RxDataset(te[ORIG_FEATURES + [OUTCOME_COL, PATIENT_ID_COL]])

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    K = int(max(df_use[CAT_RX].max(), 0) + 1)

    model = F2RxHead(in_dim=len(ORIG_FEATURES), n_cont=len(CONT_RX), n_cat=K).to(DEVICE)

    # Stage A
    train_stage_A(model, dl_tr, dl_va)
    stateA = os.path.join(REPORT_DIR, "f2_stageA_best.pth")
    model.load_state_dict(torch.load(stateA, map_location=DEVICE, weights_only=True))

    # Stage B
    train_stage_B(model, dl_tr, dl_va, teacher_state_path=stateA, num_classes=K)

    # Inference on val/test with hard constraints
    rx_val = infer_with_constraints(model, va[ORIG_FEATURES])
    rx_test = infer_with_constraints(model, te[ORIG_FEATURES])

    # Evaluate (prints to terminal, saves only plots)
    evaluate_and_plots(va, rx_val, "val")
    evaluate_and_plots(te, rx_test, "test")

    # Export predicted prescriptions (still keep CSV, as before)
    rx_val.to_csv(os.path.join(REPORT_DIR, "val_model_rx.csv"), index=False)
    rx_test.to_csv(os.path.join(REPORT_DIR, "test_model_rx.csv"), index=False)
    print(f"\nSaved predicted prescriptions to: {REPORT_DIR}/val_model_rx.csv and {REPORT_DIR}/test_model_rx.csv")
    print("All plots (scatter, confusion, loss curves) saved to:", REPORT_DIR)

if __name__ == "__main__":
    main()
