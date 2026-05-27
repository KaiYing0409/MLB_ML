import numpy as np
import pandas as pd

# ============================================================
# 設定區
# ============================================================
INPUT_PATH  = 'testdata_only_phy.csv'      # Step 1 產出（無鏡像版）
OUTPUT_PATH = 'testdata_relative_phy.csv'  # Step 3 的輸入

# ============================================================
# STEP 1：讀取資料
# ============================================================
print("===== STEP 1：讀取原始資料 =====")
df = pd.read_csv(INPUT_PATH)
print(f"原始資料筆數：{len(df):,} 筆")

# ============================================================
# STEP 2：過濾樣本數過低的邊緣人投手
# ============================================================
print("\n===== STEP 2：過濾邊緣人投手（< 50 球）=====")
MIN_PITCHES_THRESHOLD = 50
pitcher_counts = df['pitcher'].value_counts()
low_sample_pitchers = pitcher_counts[pitcher_counts < MIN_PITCHES_THRESHOLD].index
df = df[~df['pitcher'].isin(low_sample_pitchers)].reset_index(drop=True)
print(f"移除 {len(low_sample_pitchers)} 位投手，剩餘：{len(df):,} 筆")

# ============================================================
# STEP 3：spin_axis 週期性邊界轉換
# ============================================================
print("\n===== STEP 3：spin_axis → sin/cos 轉換 =====")
# spin_axis 是 0–360° 的循環變數，直接使用會有邊界突變（5° 和 355° 數值差 350° 但實際只差 10°）
# 轉換為二維向量消除此問題
df['spin_axis_rad'] = np.radians(df['spin_axis'])
df['spin_axis_sin'] = np.sin(df['spin_axis_rad'])
df['spin_axis_cos'] = np.cos(df['spin_axis_rad'])
print("  已新增：spin_axis_sin, spin_axis_cos")

# ============================================================
# STEP 4：儲存
# ============================================================
print("\n===== STEP 4：儲存 =====")
df.to_csv(OUTPUT_PATH, index=False)
print(f"已儲存：{OUTPUT_PATH}（{len(df):,} 筆）")
print("\n進入分類器的特徵：")
print("  位移：api_break_x_arm, api_break_z_with_gravity, pfx_x")
print("  速度：release_speed, release_spin_rate")
print("  加速度：ax, ay, vx0, vy0")
print("  旋轉：spin_axis_sin, spin_axis_cos")
print("  出手：arm_angle")
