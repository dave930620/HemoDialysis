import pandas as pd

# ============== 路徑設定 ==============
PRESC_CSV = "all_prescriptions.csv"
PATIENT_CSV = "final_processed_data.csv"
CONFLICT_OUT = "duplicate_conflicts.csv"
CLEANED_OUT = "cleaned_data_after_fill_and_drop.csv"

# ============== 小工具 ==============
def get_num_col(df, colname):
    """安全取欄位並轉成數值，不存在或轉不動→0"""
    if colname not in df.columns:
        return pd.Series(0, index=df.index, dtype="float64")
    return pd.to_numeric(df[colname], errors="coerce").fillna(0.0)

def to_py_int(x):
    """轉成純 Python int（避免印出 np.int32/64 樣式），失敗則原樣回傳"""
    try:
        return int(x)
    except Exception:
        return x

# ============== 讀檔 ==============
prescriptions = pd.read_csv(PRESC_CSV, encoding="utf-8-sig")
patients = pd.read_csv(PATIENT_CSV, encoding="utf-8")

# ============== 時間欄處理 ==============
# 處方：由「處方日期」取 年/月
prescriptions["處方日期"] = pd.to_datetime(prescriptions["處方日期"], errors="coerce")
prescriptions["年"] = prescriptions["處方日期"].dt.year
prescriptions["月"] = prescriptions["處方日期"].dt.month

# 病人：由「記錄時間」取 年/月
patients["記錄時間"] = pd.to_datetime(patients["記錄時間"], errors="coerce")
patients["年"] = patients["記錄時間"].dt.year
patients["月"] = patients["記錄時間"].dt.month

# ============== 計算 6 個新欄位（空值補 0） ==============
# 日間 1~4 組：濃度_i * 每袋升數_i * 每日袋數_i
glucose_cols, calcium_cols = [], []
for i in range(1, 5):
    g = get_num_col(prescriptions, f"葡萄糖濃度_{i}")
    c = get_num_col(prescriptions, f"鈣離子濃度_{i}")
    vol = get_num_col(prescriptions, f"每袋升數_{i}")
    bags = get_num_col(prescriptions, f"每日袋數_{i}")
    glucose_cols.append(g * vol * bags)
    calcium_cols.append(c * vol * bags)

# 夜間 N1~N3：以 APD換液升數_i * APD換液次數_i 當體積
night_glucose_cols, night_calcium_cols = [], []
apd_vol_cols, apd_cnt_cols = [], []
for i in range(1, 4):
    apd_vol_i = get_num_col(prescriptions, f"APD換液升數_{i}")
    apd_vol_cols.append(apd_vol_i)

    apd_cnt_i = get_num_col(prescriptions, f"APD換液次數_{i}")
    if i == 3 and "APDL換液次數_3" in prescriptions.columns:
        apd_cnt_i = apd_cnt_i + get_num_col(prescriptions, "APDL換液次數_3")
    apd_cnt_cols.append(apd_cnt_i)

    g_n = get_num_col(prescriptions, f"葡萄糖濃度_N{i}")
    c_n = get_num_col(prescriptions, f"鈣離子濃度_N{i}")
    night_volume_i = apd_vol_i * apd_cnt_i
    night_glucose_cols.append(g_n * night_volume_i)
    night_calcium_cols.append(c_n * night_volume_i)

# 六個新欄位（英文）
six_cols = [
    "glucose_total", "calcium_total",
    "glucose_total_n", "calcium_total_n",
    "apd_total_volume", "apd_total_cycles"
]
prescriptions["glucose_total"]    = sum(glucose_cols)
prescriptions["calcium_total"]    = sum(calcium_cols)
prescriptions["glucose_total_n"]  = sum(night_glucose_cols)
prescriptions["calcium_total_n"]  = sum(night_calcium_cols)
prescriptions["apd_total_volume"] = sum(apd_vol_cols)
prescriptions["apd_total_cycles"] = sum(apd_cnt_cols)

# 空值補 0（保證六欄不為 NaN）
prescriptions[six_cols] = prescriptions[six_cols].fillna(0.0)

# ============== 只保留合併需要的欄位（key + 6 欄） ==============
key_cols = ["PatientID", "年", "月"]
prescriptions_slim = prescriptions[key_cols + six_cols].copy()

# ============== 檢查並處理重複 (PatientID, 年, 月) ==============
dup_mask = prescriptions_slim.duplicated(subset=key_cols, keep=False)
dup_df = prescriptions_slim[dup_mask].copy()

if not dup_df.empty:
    conflict_keys = []
    for (pid, y, m), group in dup_df.groupby(key_cols, dropna=False):
        # 六欄全部一致才算一致
        all_same = (group[six_cols].nunique(dropna=False) == 1).all()
        key_tuple = (to_py_int(pid), to_py_int(y), to_py_int(m))
        if not all_same:
            conflict_keys.append(key_tuple)

    if conflict_keys:
        # 輸出衝突名單
        conflict_df = pd.DataFrame(conflict_keys, columns=["PatientID", "年", "月"]).drop_duplicates()
        conflict_df.to_csv(CONFLICT_OUT, index=False, encoding="utf-8-sig")
        print(f"⚠️ 發現 {len(conflict_df)} 個 (PatientID, 年, 月) 的六欄數值不一致，已輸出：{CONFLICT_OUT}")

        # 從 prescriptions_slim 刪掉整組衝突 key
        pres_keys = list(zip(
            prescriptions_slim["PatientID"].map(to_py_int),
            prescriptions_slim["年"].map(to_py_int),
            prescriptions_slim["月"].map(to_py_int),
        ))
        prescriptions_slim = prescriptions_slim.loc[
            [k not in set(map(tuple, conflict_df.values)) for k in pres_keys]
        ].copy()
    else:
        print("✅ 重複的 key 其六欄數值皆一致（將合併為一行）。")

    # 對於一致的重複，僅保留一列
    prescriptions_slim = prescriptions_slim.drop_duplicates(subset=key_cols, keep="first")
else:
    print("✅ 沒有 (PatientID, 年, 月) 的重複資料。")

# ============== 與病人資料合併（只帶入 6 欄）、LOCF、清理 ==============
merged = pd.merge(
    patients,
    prescriptions_slim,
    how="left",
    on=key_cols,
    suffixes=("", "_處方")
)

# 只針對這 6 欄做 LOCF
merged = merged.sort_values(by=["PatientID", "記錄時間"])
merged[six_cols] = merged.groupby("PatientID", dropna=False)[six_cols].ffill()

# 刪除仍有缺失值的 row（僅依 6 欄判定）
merged_dropped = merged.dropna(subset=six_cols, how="any")

# ============== 檢查輸出 & 型別檢核 ==============
print(f"✅ 合併後（LOCF+清理）共有 {len(merged_dropped)} 行資料")

# 檢查非數值欄位（允許：日期欄與 PatientID）
non_numeric_cols = []
for col in merged_dropped.columns:
    if pd.api.types.is_numeric_dtype(merged_dropped[col]):
        continue
    if pd.api.types.is_datetime64_any_dtype(merged_dropped[col]):
        continue
    if col.lower() in ["patientid"]:
        continue
    non_numeric_cols.append(col)

if non_numeric_cols:
    print("\n⚠️ 以下欄位不是數值或日期（可能含文字/分類資料，或格式未轉數字）：")
    for col in non_numeric_cols:
        sample_vals = merged_dropped[col].dropna().unique()[:5]
        sample_vals = [str(v)[:50] for v in sample_vals]
        print(f" - {col} (前5個值: {sample_vals})")
else:
    print("\n✅ 除了日期與 PatientID，所有欄位皆為數值")

# ============== 存檔 ==============
merged_dropped.to_csv(CLEANED_OUT, index=False, encoding="utf-8-sig")
print(f"✅ 已輸出：{CLEANED_OUT}")
