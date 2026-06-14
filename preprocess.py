import numpy as np
import pandas as pd

DATA_PATH    = 'statcast_bat_tracking_2024_2025.csv'
OUT_FULL     = 'testdata_only_phy.csv'
OUT_PHY      = 'Pitch_physical_only.csv'
OUT_TRAIN    = 'data_train.csv'
OUT_VAL      = 'data_val.csv'
OUT_TEST     = 'data_test.csv'

N_SAMPLE  = 100_000
K         = 3.0
BIN_DEG   = 10
MIN_RATIO = 0.01
SEED      = 42

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

MODEL_FEATS = [
    'release_speed', 'release_spin_rate', 'spin_axis',
    'pfx_x', 'api_break_z_with_gravity',
    'vx0', 'vz0', 'ay',
]

print("載入資料中...")
df = pd.read_csv(DATA_PATH, usecols=lambda c: c in PHYSICAL_COLS)
print(f"原始資料：{len(df):,} 筆，{df.shape[1]} 欄")


# STEP 2：基本清理
df = df.dropna(subset=['pitch_type'])

counts      = df['pitch_type'].value_counts()
valid_types = counts[counts / len(df) >= MIN_RATIO].index
df          = df[df['pitch_type'].isin(valid_types)].copy()
print(f"過濾低佔比球種後：{len(df):,} 筆，保留球種：{sorted(valid_types.tolist())}")

# STEP 3：KC → CU 合併
df['pitch_type'] = df['pitch_type'].replace('KC', 'CU')
print(f"KC → CU 合併後：{len(df):,} 筆")


# STEP 4：IQR 離群移除
def circular_iqr_outlier_mask(series, k=3.0, bin_deg=10):
    """spin_axis 專用：循環感知 IQR 離群偵測。"""
    bins  = np.arange(0, 361, bin_deg)
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
    """對每個（pitch_type × p_throws）子群做 IQR 離群移除。"""
    num_cols     = df.select_dtypes(include=np.number).columns.tolist()
    general_cols = [c for c in num_cols if c != 'spin_axis']
    outlier_idx  = set()

    for (pt, hand), grp in df.groupby(['pitch_type', 'p_throws']):
        mask = pd.Series(False, index=grp.index)

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

        if 'spin_axis' in grp.columns:
            s = grp['spin_axis'].dropna()
            if len(s) >= 4:
                circ_mask = circular_iqr_outlier_mask(s, k=k, bin_deg=bin_deg)
                mask.loc[circ_mask[circ_mask].index] = True

        outlier_idx.update(grp.index[mask].tolist())

    n_before  = len(df)
    df_clean  = df.drop(index=list(outlier_idx)).reset_index(drop=True)
    print(f"IQR 離群移除：{len(outlier_idx):,} 筆 "
          f"({len(outlier_idx)/n_before:.2%})，"
          f"剩餘：{len(df_clean):,} 筆")
    return df_clean


df = remove_outliers(df, k=K, bin_deg=BIN_DEG)

# STEP 5：移除模型特徵缺值
n_before = len(df)
df = df.dropna(subset=MODEL_FEATS)
print(f"移除特徵缺值後：{n_before - len(df):,} 筆移除，剩餘：{len(df):,} 筆")

# STEP 6：spin_axis → sin/cos 轉換
df['spin_axis_sin'] = np.sin(np.deg2rad(df['spin_axis']))
df['spin_axis_cos'] = np.cos(np.deg2rad(df['spin_axis']))
print("spin_axis sin/cos 轉換完成")

# STEP 7：儲存 IQR 清理後完整資料 → Pitch_physical_only.csv
df.to_csv(OUT_PHY, index=False)
print(f"\n已儲存（IQR清理後完整資料）：{OUT_PHY}（{len(df):,} 筆）")

# STEP 8：分層抽樣 10 萬筆 → testdata_only_phy.csv
def stratified_sample(data, n, seed=42):
    frames = []
    for pt, grp in data.groupby('pitch_type'):
        k = min(len(grp), int(n * len(grp) / len(data)))
        frames.append(grp.sample(n=k, random_state=seed))
    return pd.concat(frames).reset_index(drop=True)

df_sample = stratified_sample(df, N_SAMPLE, seed=SEED)
print(f"\n分層抽樣後：{len(df_sample):,} 筆")
print(df_sample['pitch_type'].value_counts().to_string())

df_sample.to_csv(OUT_FULL, index=False)
print(f"\n已儲存：{OUT_FULL}")

# STEP 9：切分 60/20/20 → data_train/val/test.csv
rng = np.random.RandomState(SEED)
idx = np.arange(len(df_sample))
rng.shuffle(idx)

n     = len(idx)
n_tr  = int(n * 0.6)
n_va  = int(n * 0.2)

df_train = df_sample.iloc[idx[:n_tr]].reset_index(drop=True)
df_val   = df_sample.iloc[idx[n_tr:n_tr + n_va]].reset_index(drop=True)
df_test  = df_sample.iloc[idx[n_tr + n_va:]].reset_index(drop=True)

df_train.to_csv(OUT_TRAIN, index=False)
df_val.to_csv(OUT_VAL,   index=False)
df_test.to_csv(OUT_TEST,  index=False)

print(f"\n資料切分完成（seed={SEED}）：")
print(f"  Train：{len(df_train):,} 筆 → {OUT_TRAIN}")
print(f"  Val  ：{len(df_val):,} 筆 → {OUT_VAL}")
print(f"  Test ：{len(df_test):,} 筆 → {OUT_TEST}")
