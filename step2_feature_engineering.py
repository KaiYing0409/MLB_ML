import numpy as np
import pandas as pd

# ============================================================
# 1. 設定區
# ============================================================
INPUT_PATH = 'testdata_only_phy.csv'       # Step 1 產出的【無鏡像版】
# ⚠️  注意：必須用無鏡像版，不能用 testdata_only_phy_mirror.csv
#    原因：相對特徵以各投手 FF 基準線計算，若輸入已鏡像的資料
#          左投水平量已被翻轉，基準線會失真，導致後續 LDA 左右投準確率崩潰
OUTPUT_PATH = 'testdata_relative_phy.csv'   # 轉換後的相對特徵資料集（Step 3 的輸入）

# 你目前 KNN/QDA 模型挑選出的核心物理特徵清單
# 註：ay 屬於加速度，spin_axis 是角度（在下面會自動做 sin/cos 轉換）
# 我們要對速度與位移這四大絕對物理量計算「相對速球差」
ABS_FEATURES = [
    'release_speed', 
    'release_spin_rate', 
    'api_break_x_arm', 
    'api_break_z_with_gravity'
]

# 邊緣人投手過濾門檻：兩年內（2024-2025）總投球數低於此值的投手直接剔除
MIN_PITCHES_THRESHOLD = 50 

print("===== STEP 1：讀取原始資料 =====")
df = pd.read_csv(INPUT_PATH)
n_original = len(df)
print(f"原始資料筆數：{n_original:,} 筆")

# ============================================================
# 2. 砍掉邊緣人投手（資料清洗）
# ============================================================
print("\n===== STEP 2：過濾樣本數過低的邊緣人投手 =====")
pitcher_counts = df['pitcher'].value_counts()
low_sample_pitchers = pitcher_counts[pitcher_counts < MIN_PITCHES_THRESHOLD].index

df_cleaned = df[~df['pitcher'].isin(low_sample_pitchers)].reset_index(drop=True)
n_after_filter = len(df_cleaned)

print(f"被移除的邊緣人投手人數：{len(low_sample_pitchers)} 人")
print(f"移除邊緣人後剩餘資料：{n_after_filter:,} 筆（砍掉了 {n_original - n_after_filter:,} 筆）")

# ============================================================
# 3. 計算基準線（Baseline Dictionary）
# ============================================================
print("\n===== STEP 3：計算各投手的四縫線速球 (FF) 基準線 =====")

# 全聯盟的 FF 平均值（兜底防線：當某投手完全沒投過 FF 時使用）
df_ff_global = df_cleaned[df_cleaned['pitch_type'] == 'FF']
global_ff_baseline = df_ff_global[ABS_FEATURES].mean()

print("全聯盟四縫線速球 (FF) 全局均值（作為防守兜底）：")
for feat in ABS_FEATURES:
    print(f"  {feat}: {global_ff_baseline[feat]:.2f}")

# 分組計算「每個投手」的 FF 平均值
pitcher_ff_groups = df_ff_global.groupby('pitcher')[ABS_FEATURES].mean()

# ============================================================
# 4. 生成相對特徵（向量化高效運算）
# ============================================================
print("\n===== STEP 4：進行以投手為基準的特徵對齊 =====")

# 建立對照字典，方便快速 Lookup
# 結構為 {pitcher_id: {feat1: mean1, feat2: mean2, ...}}
baseline_dict = pitcher_ff_groups.to_dict(orient='index')

# 建立儲存新特徵的 DataFrame
df_rel = df_cleaned.copy()

# 初始化儲存相對特徵的矩陣，加快 Pandas 寫入速度
rel_matrix = {f'rel_{feat}': np.zeros(len(df_rel)) for feat in ABS_FEATURES}

# 遍歷每一筆投球資料，計算相對值
# 利用 values 的 numpy 陣列進行快取，優化迴圈效能
pitchers = df_rel['pitcher'].values
abs_data = {feat: df_rel[feat].values for feat in ABS_FEATURES}

no_ff_count = 0
used_pitchers_with_no_ff = set()

for i in range(len(df_rel)):
    p_id = pitchers[i]
    
    # 如果該投手有自己的 FF 基準線，就用他的；否則代入全聯盟均值
    if p_id in baseline_dict:
        baseline = baseline_dict[p_id]
    else:
        baseline = global_ff_baseline
        if p_id not in used_pitchers_with_no_ff:
            no_ff_count += 1
            used_pitchers_with_no_ff.add(p_id)
            
    # 計算相對差值
    for feat in ABS_FEATURES:
        rel_matrix[f'rel_{feat}'][i] = abs_data[feat][i] - baseline[feat]

print(f"  註：有 {no_ff_count} 位總球數達標、但這兩年完全沒丟過 FF 的特殊型投手，已自動補正為全聯盟均值。")

# 將計算好的相對特徵塞回 DataFrame
for col_name, data_array in rel_matrix.items():
    df_rel[col_name] = data_array

# ============================================================
# 5. 進階特徵工程：旋轉軸角度 (spin_axis) 週期性邊界轉換
# ============================================================
print("\n===== STEP 5：將角度特徵 spin_axis 轉換為二維正餘弦軌道 =====")
# 修正角度在 0°/360° 連續變數計算距離時的突變盲點
df_rel['spin_axis_rad'] = np.radians(df_rel['spin_axis'])
df_rel['spin_axis_sin'] = np.sin(df_rel['spin_axis_rad'])
df_rel['spin_axis_cos'] = np.cos(df_rel['spin_axis_rad'])

# ============================================================
# 6. 輸出成果與後續對照
# ============================================================
print("\n===== STEP 6：儲存對齊後的全新資料集 =====")
df_rel.to_csv(OUTPUT_PATH, index=False)
print(f"成功儲存！新檔案路徑：{OUTPUT_PATH}")

print("\n【特徵工程完成！你的全新特徵組合建議】")
print("原先訓練 KNN/QDA 的特徵：")
print("  ['api_break_x_arm', 'api_break_z_with_gravity', 'spin_axis', 'release_speed', 'release_spin_rate', 'ay']")
print("\n請在下一輪的模型腳本中，將特徵清單（FINAL_FEATURES）替換成這組『對齊後的特徵』：")
print("  ['rel_api_break_x_arm',")
print("   'rel_api_break_z_with_gravity',")
print("   'spin_axis_sin',")
print("   'spin_axis_cos',")
print("   'rel_release_speed',")
print("   'rel_release_spin_rate',")
print("   'ay']")
print("============================================================")