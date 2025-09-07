
import pandas as pd



# 讀取 CSV 檔案
df_final = pd.read_csv('final_processed_data.csv')
df_all = pd.read_csv('all_prescriptions.csv')

# 確保欄位名稱一致、轉小寫
df_final.columns = [col.lower() for col in df_final.columns]
df_all.columns = [col.lower() for col in df_all.columns]

# 取得 patientId 並去除空值與重複
final_ids = set(df_final['patientid'].dropna().astype(str).unique())
all_ids = set(df_all['patientid'].dropna().astype(str).unique())
print(f"✅ final_processed_data.csv 出現的病人數量：{len(final_ids)}")
print(f"✅ all_prescriptions.csv 出現的病人數量：{len(all_ids)}")

# 比較
only_in_final = final_ids - all_ids
only_in_all = all_ids - final_ids

# 輸出
print(f"✅ 只在 final_processed_data.csv 出現的病人數量：{len(only_in_final)}")
print("病人 ID：", sorted(only_in_final))

print(f"\n✅ 只在 all_prescriptions.csv 出現的病人數量：{len(only_in_all)}")
print("病人 ID：", sorted(only_in_all))
pd.DataFrame(sorted(only_in_final), columns=['patientId']).to_csv('patientLoss.csv', index=False)
#pd.DataFrame(sorted(only_in_all), columns=['patientId']).to_csv('only_in_all.csv', index=False)

"""
import pandas as pd

# 讀取資料
df = pd.read_csv('final_processed_data.csv')

# 查看總筆數
total_count = len(df)

# 根據條件篩選
count_ge_1_7 = (df['PD Kt/V'] >= 1.7).sum()
count_lt_1_7 = (df['PD Kt/v'] < 1.7).sum()

# 輸出結果
print(f"總共有 {total_count} 筆資料")
print(f"'PD kt/v' >= 1.7 的筆數：{count_ge_1_7}")
print(f"'PD kt/v' < 1.7 的筆數：{count_lt_1_7}")
"""
