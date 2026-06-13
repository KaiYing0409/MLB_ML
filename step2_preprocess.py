"""
MLB Statcast 投球資料 — 前處理
================================
輸入：statcast_bat_tracking_2024_2025.csv
輸出：testdata_only_phy.csv（10 萬筆，乾淨資料）

處理步驟：
  1. 載入 & 選欄位
  2. 去 pitch_type 缺值
  3. 過濾佔比 < 1% 的球種（決定分類目標）
  4. KC → CU 合併（第一版分類器迭代後的決策，詳見 KC_CU_合併理由.md）
  5. IQR 離群移除（按球種×手性分組）
     - 一般特徵：Q1-3IQR ~ Q3+3IQR
     - spin_axis：循環感知平移法（避免 0°/360° 邊界誤刪）
  6. 分層抽樣 10 萬筆（seed=42）
  7. 輸出
"""

import numpy as np
import pandas as pd

DATA_PATH   = 'statcast_bat_tracking_2024_2025.csv'
OUTPUT_PATH_10W = 'testdata_only_phy.csv'
OUTPUT_PATH_COACH = 'Pitch_physical_only.csv' # 這就是給 AI 教練用的
N_SAMPLE    = 100_000
K           = 3.0
BIN_DEG     = 10
MIN_RATIO   = 0.01
SEED        = 42

PHYSICAL_COLS = [
    'pitch_type', 'pitch_name',
    'release_speed', 'effective_speed',
    'release_spin_rate', 'spin_axis',
    'pfx_x', 'pfx_z',
    'api_break_z_with_gravity',
    'api_break_x_arm', 'api_break_x_batter_in',
    'release_pos_x', 'release_pos_z', 'release_pos_y',
    'release_extension', 'arm_angle',
    'plate_x', 'plate_z', 'zone',
    'vx0', 'vy0', 'vz0',
    'ax', 'ay', 'az',
    'sz_top', 'sz_bot',
    'p_throws', 'stand', 'pitcher',
]

# ============================================================
# STEP 1：載入
# ============================================================
print("載入資料中...")
df = pd.read_csv(DATA_PATH, usecols=lambda c: c in PHYSICAL_COLS)
print(f"原始資料：{len(df):,} 筆，{df.shape[1]} 欄")

# ============================================================
# STEP 2：基本清理
# ============================================================
df = df.dropna(subset=['pitch_type'])

# 過濾佔比 < MIN_RATIO 的球種（決定分類目標，在合併之前）
counts      = df['pitch_type'].value_counts()
valid_types = counts[counts / len(df) >= MIN_RATIO].index
df          = df[df['pitch_type'].isin(valid_types)].copy()
print(f"過濾低佔比球種後：{len(df):,} 筆，保留球種：{sorted(valid_types.tolist())}")

# ============================================================
# STEP 3：KC → CU 合併
# （第一版分類器觀察到 KC 幾乎全被判成 CU，
#   驗證物理特性後確認兩者 spin_axis 分布完全重疊，詳見 KC_CU_合併理由.md）
# ============================================================
df['pitch_type'] = df['pitch_type'].replace('KC', 'CU')
print(f"KC → CU 合併後：{len(df):,} 筆")

# ============================================================
# STEP 4：IQR 離群移除（按球種×手性分組，k=3）
# ============================================================

def circular_iqr_outlier_mask(series, k=3.0, bin_deg=10):
    """
    spin_axis 專用：循環感知 IQR 離群偵測。
    1. 用 histogram（bin=bin_deg°）找眾數角度
    2. 平移使眾數落在 180°（遠離 0°/360° 邊界）
    3. 在平移空間做 IQR，找離群樣本
    回傳：布林 mask，True = 離群
    """
    bins   = np.arange(0, 361, bin_deg)
    counts, edges = np.histogram(series.dropna(), bins=bins)
    mode_idx   = np.argmax(counts)
    mode_angle = (edges[mode_idx] + edges[mode_idx + 1]) / 2

    offset  = 180.0 - mode_angle
    shifted = (series + offset) % 360

    q1  = shifted.quantile(0.25)
    q3  = shifted.quantile(0.75)
    iqr = q3 - q1
    lo  = q1 - k * iqr
    hi  = q3 + k * iqr

    return (shifted < lo) | (shifted > hi)


def remove_outliers(df, k=3.0, bin_deg=10):
    """
    對每個（pitch_type × p_throws）子群做 IQR 離群移除。
    spin_axis 用循環感知版本，其他數值特徵用一般 IQR。
    回傳：清理後的 DataFrame。
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    # spin_axis 單獨處理，不放進一般 IQR
    general_cols = [c for c in num_cols if c != 'spin_axis']

    outlier_idx = set()

    for (pt, hand), grp in df.groupby(['pitch_type', 'p_throws']):
        mask = pd.Series(False, index=grp.index)

        # 一般特徵：標準 IQR
        for col in general_cols:
            s = grp[col].dropna()
            if len(s) < 4:
                continue
            q1  = s.quantile(0.25)
            q3  = s.quantile(0.75)
            iqr = q3 - q1
            lo  = q1 - k * iqr
            hi  = q3 + k * iqr
            mask |= grp[col].notna() & ((grp[col] < lo) | (grp[col] > hi))

        # spin_axis：循環感知
        if 'spin_axis' in grp.columns:
            s = grp['spin_axis'].dropna()
            if len(s) >= 4:
                circ_mask = circular_iqr_outlier_mask(s, k=k, bin_deg=bin_deg)
                # circ_mask index 對應 s.index，需對齊回 grp.index
                mask.loc[circ_mask[circ_mask].index] = True

        outlier_idx.update(grp.index[mask].tolist())

    n_before = len(df)
    df_clean = df.drop(index=list(outlier_idx)).reset_index(drop=True)
    print(f"IQR 離群移除：{len(outlier_idx):,} 筆 "
          f"({len(outlier_idx)/n_before:.2%})，"
          f"剩餘：{len(df_clean):,} 筆")
    return df_clean


df = remove_outliers(df, k=K, bin_deg=BIN_DEG)

# ============================================================
# STEP 5：移除模型特徵欄位有缺值的筆數
# （IQR 只移除有值但超出範圍的樣本，缺值需另外處理）
# ============================================================
MODEL_FEATS = [
    'release_speed', 'release_spin_rate', 'spin_axis',
    'pfx_x', 'api_break_z_with_gravity',
    'vx0', 'vz0', 'ay',
]
n_before = len(df)
df = df.dropna(subset=MODEL_FEATS)
print(f"移除特徵缺值後：{n_before - len(df):,} 筆移除，剩餘：{len(df):,} 筆")

# ============================================================
# STEP 6：spin_axis → sin/cos 轉換
# （spin_axis 為循環變數，直接使用角度值會有 0°/360° 邊界問題，
#   轉換為歐氏空間的線性特徵，原始欄位保留供參考）
# ============================================================
df['spin_axis_sin'] = np.sin(np.deg2rad(df['spin_axis']))
df['spin_axis_cos'] = np.cos(np.deg2rad(df['spin_axis']))
print(f"spin_axis sin/cos 轉換完成")

COACH_DB_PATH = 'Pitch_physical_only.csv'
df.to_csv(COACH_DB_PATH, index=False)
print(f"\n已儲存 AI 教練專用大母體資料庫：{COACH_DB_PATH} (共 {len(df):,} 筆)")
# ============================================================
# STEP 7：分層抽樣 10 萬筆
# ============================================================
def stratified_sample(data, n, seed=42):
    frames = []
    for pt, grp in data.groupby('pitch_type'):
        k = min(len(grp), int(n * len(grp) / len(data)))
        frames.append(grp.sample(n=k, random_state=seed))
    return pd.concat(frames).reset_index(drop=True)

df_sample = stratified_sample(df, N_SAMPLE, seed=SEED)
print(f"\n分層抽樣後：{len(df_sample):,} 筆")
print(df_sample['pitch_type'].value_counts().to_string())

# ============================================================
# STEP 8：輸出
# ============================================================
# 1. 先輸出 AI 教練專用的大母體資料庫 (138 萬筆乾淨版)
df.to_csv(OUTPUT_PATH_COACH, index=False)
print(f"\n✅ 已儲存 AI 教練專用大母體資料庫：{OUTPUT_PATH_COACH} (共 {len(df):,} 筆)")

# 2. 再輸出分類器模型訓練用的分層抽樣資料庫 (10 萬筆平衡版)
df_sample.to_csv(OUTPUT_PATH_10W, index=False)
print(f"✅ 已儲存分類器專用抽樣資料庫：{OUTPUT_PATH_10W} (共 {len(df_sample):,} 筆)")
