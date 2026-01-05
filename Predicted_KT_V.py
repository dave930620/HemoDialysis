"""Predicted_KT_V

Outcome prediction model (F1): predicts PD adequacy metrics such as Kt/V from structured
tabular inputs.

This file is intended to be a single, self-contained script that:
- Loads a study CSV
- Preprocesses discrete/continuous features (imputation + scaling)
- Trains a SAINT-like transformer model for regression
- Evaluates on a held-out split and produces diagnostic plots
- Saves a PyTorch checkpoint and (optionally) feature-importance outputs

Assumptions
- You have a local CSV file with the same column names used during the study.
- Any paths, column lists, or hyperparameters are configured inside `main()`.

Open-source tips
- Do not commit private datasets. Use `.gitignore` for `data/` and `checkpoints/`.
- Provide a small demo CSV (synthetic or anonymized) so others can run end-to-end.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


# ==================== Data preprocessing ====================
def preprocess_data(data):
    """Preprocess a raw pandas DataFrame into model-ready tensors.

    Steps
    - Select discrete and continuous feature columns
    - Impute missing values
    - Standardize continuous features
    - Build torch tensors for training

    Returns
    - X: torch.FloatTensor, shape (N, D)
    - y: torch.FloatTensor, shape (N, 1)
    - preprocessors used (imputer/scaler) when the function fits them
    """
    data = data.copy()

    if "PatientID" in data.columns:
        data["PatientID"] = data["PatientID"].astype(str)

    discrete_features = [
        "Gender",
        "HBsAg",
        "Anti-HCV",
        "Comorbidities",
        "Dialysis_method",
        "Anticoagulant",
        "Base",
        "Dialyzer_model",
    ]
    continuous_features = [
        "Age",
        "Height",
        "Weight",
        "BSA",
        "BP_systolic",
        "BP_diastolic",
        "BUN_pre",
        "BUN_post",
        "Creatinine",
        "Albumin",
        "Total_protein",
        "Hemoglobin",
        "Hematocrit",
        "WBC",
        "Platelet",
        "Sodium",
        "Potassium",
        "Calcium",
        "Phosphorus",
        "Chloride",
        "Bicarbonate",
        "Dialysis_time",
        "Blood_flow_rate",
        "Dialysate_flow_rate",
        "UF_volume",
        "KtV_Gotch",
        "URR",
    ]

    target_column = "PD_KtV"

    for col in discrete_features + continuous_features + [target_column]:
        if col not in data.columns:
            data[col] = np.nan

    def to_numeric_safely(series):
        return pd.to_numeric(series, errors="coerce")

    for col in continuous_features + [target_column]:
        data[col] = to_numeric_safely(data[col])

    for col in discrete_features:
        data[col] = data[col].astype(str).fillna("nan")

    X_discrete = data[discrete_features].copy()
    X_cont = data[continuous_features].copy()
    y = data[[target_column]].copy()

    cont_imputer = SimpleImputer(strategy="median")
    X_cont_imputed = cont_imputer.fit_transform(X_cont)

    scaler = StandardScaler()
    X_cont_scaled = scaler.fit_transform(X_cont_imputed)

    def encode_discrete_df(df):
        encoded = []
        encoders = {}
        for col in df.columns:
            categories = sorted(df[col].astype(str).unique().tolist())
            mapping = {c: i for i, c in enumerate(categories)}
            encoders[col] = mapping
            encoded.append(df[col].astype(str).map(mapping).fillna(0).astype(int).values.reshape(-1, 1))
        return np.concatenate(encoded, axis=1), encoders

    X_disc_encoded, disc_encoders = encode_discrete_df(X_discrete)

    X = np.concatenate([X_disc_encoded.astype(np.float32), X_cont_scaled.astype(np.float32)], axis=1)

    y_imputer = SimpleImputer(strategy="median")
    y_imputed = y_imputer.fit_transform(y).astype(np.float32)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_imputed, dtype=torch.float32)

    preprocessors = {
        "discrete_features": discrete_features,
        "continuous_features": continuous_features,
        "target_column": target_column,
        "disc_encoders": disc_encoders,
        "cont_imputer": cont_imputer,
        "cont_scaler": scaler,
        "y_imputer": y_imputer,
        "num_discrete": len(discrete_features),
        "num_continuous": len(continuous_features),
    }
    return X_tensor, y_tensor, preprocessors


# ==================== Model definition ====================
class SAINT(nn.Module):
    """SAINT-style transformer model for mixed discrete/continuous tabular features.

    High-level architecture
    - Discrete features -> embeddings
    - Continuous features -> linear projection
    - Concatenate into a sequence and encode with TransformerEncoder layers
    - Pool and regress to a single output
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        discrete_feature_indices,
        continuous_feature_indices,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
    ):
        super(SAINT, self).__init__()

        self.hidden_size = hidden_size
        self.discrete_feature_indices = discrete_feature_indices
        self.continuous_feature_indices = continuous_feature_indices

        self.num_discrete = len(discrete_feature_indices)
        self.num_continuous = len(continuous_feature_indices)

        self.discrete_embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=1000, embedding_dim=hidden_size) for _ in range(self.num_discrete)]
        )

        self.continuous_projection = nn.Linear(self.num_continuous, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        batch_size = x.size(0)

        discrete_parts = []
        for i, idx in enumerate(self.discrete_feature_indices):
            xi = x[:, idx].long().clamp(min=0)
            discrete_parts.append(self.discrete_embeddings[i](xi).unsqueeze(1))

        disc_tokens = torch.cat(discrete_parts, dim=1) if len(discrete_parts) > 0 else None

        cont_x = x[:, self.continuous_feature_indices]
        cont_token = self.continuous_projection(cont_x).unsqueeze(1)

        if disc_tokens is None:
            tokens = cont_token
        else:
            tokens = torch.cat([disc_tokens, cont_token], dim=1)

        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)

        out = self.fc_out(pooled)
        return out


# ==================== Evaluation utilities ====================
def evaluate_model(model, data_loader, device="cpu"):
    """Evaluate a trained model on a DataLoader.

    Returns a dictionary with common regression metrics (MSE, RMSE, MAE, R2, MAPE when available).
    """
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            y_true_list.append(y_batch.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0).reshape(-1)
    y_pred = np.concatenate(y_pred_list, axis=0).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    try:
        metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
    except Exception:
        pass

    return metrics, y_true, y_pred


def visualize_results(y_true, y_pred, save_prefix="f1"):
    """Create diagnostic plots (e.g., y_true vs y_pred, residuals) for reporting and sanity checks."""
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("True vs Predicted")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_true_vs_pred.png")
    plt.close()

    residuals = y_true - y_pred
    plt.figure()
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual (true - pred)")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_residuals.png")
    plt.close()


def visualize_feature_importance(feature_names, importances, save_path="feature_importance.png"):
    """Compute and save/plot feature-importance estimates.

    Important: document the method (e.g., permutation importance) in your paper/repo.
    """
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(30)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=df, x="importance", y="feature")
    plt.title("Top feature importances")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return df


# ==================== Inference utility ====================
def predict_with_model(model_path, feature_info_path, csv_path, device="cpu"):
    """Load a saved model + preprocessing artifacts and run inference on a CSV.

    Returns a NumPy array of predictions.
    """
    with open(feature_info_path, "rb") as f:
        feature_info = pickle.load(f)

    df = pd.read_csv(csv_path)
    X_tensor, _, preprocessors = preprocess_data(df)

    num_discrete = preprocessors["num_discrete"]
    num_continuous = preprocessors["num_continuous"]
    discrete_feature_indices = list(range(num_discrete))
    continuous_feature_indices = list(range(num_discrete, num_discrete + num_continuous))

    hidden_size = feature_info.get("hidden_size", 128)

    model = SAINT(
        input_size=X_tensor.shape[1],
        hidden_size=hidden_size,
        output_size=1,
        discrete_feature_indices=discrete_feature_indices,
        continuous_feature_indices=continuous_feature_indices,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        preds = model(X_tensor.to(device)).cpu().numpy().reshape(-1)
    return preds


# ==================== Training entrypoint ====================
def main():
    """Run the full training pipeline.

    Edit the configuration inside this function to match your dataset paths and column names.
    """
    csv_path = "your_training_data.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(csv_path)

    X_tensor, y_tensor, preprocessors = preprocess_data(df)

    num_discrete = preprocessors["num_discrete"]
    num_continuous = preprocessors["num_continuous"]
    discrete_feature_indices = list(range(num_discrete))
    continuous_feature_indices = list(range(num_discrete, num_discrete + num_continuous))

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

    if "PatientID" in df.columns:
        groups = df["PatientID"].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(np.arange(len(df)), groups=groups))
    else:
        idx = np.arange(len(df))
        np.random.seed(42)
        np.random.shuffle(idx)
        split = int(0.8 * len(idx))
        train_idx, test_idx = idx[:split], idx[split:]

    train_set = torch.utils.data.Subset(dataset, train_idx.tolist())
    test_set = torch.utils.data.Subset(dataset, test_idx.tolist())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False)

    model = SAINT(
        input_size=X_tensor.shape[1],
        hidden_size=128,
        output_size=1,
        discrete_feature_indices=discrete_feature_indices,
        continuous_feature_indices=continuous_feature_indices,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    epochs = 30
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"[Epoch {epoch:03d}] train_loss={avg_loss:.6f}")

    metrics, y_true, y_pred = evaluate_model(model, test_loader, device=device)
    print("=== Test metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    visualize_results(y_true, y_pred, save_prefix="f1")

    torch.save(model.state_dict(), "saint_pd_model.pth")
    print("Saved model checkpoint to: saint_pd_model.pth")


if __name__ == "__main__":
    main()
