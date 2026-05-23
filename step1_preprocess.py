"""
MLB Statcast 投球資料 - 前處理與 EDA
目標：留下所有物理特徵 → 取 10 萬筆 → 找哪些特徵能把球種分開
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Noto Serif TC'    


#%%
# 所有與投球「物理」直接相關的欄位
PHYSICAL_COLS = [
    # ── 目標標籤 ──────────────────────────
    'pitch_type',           # 球種代碼（分類目標 Y）
    'pitch_name',           # 球種全名（方便畫圖用）

    # ── 球速 ──────────────────────────────
    'release_speed',        # 出手球速 (mph)
    'effective_speed',      # 有效球速（含出手延伸修正）

    # ── 旋轉 ──────────────────────────────
    'release_spin_rate',    # 轉速 (rpm)
    'spin_axis',            # 旋轉軸方向 (0-360°)

    # ── 位移（球路變化）───────────────────
    'pfx_x',                # 水平位移 (英吋，去除重力)
    'pfx_z',                # 垂直位移 (英吋，去除重力)
    'api_break_z_with_gravity',   # 含重力的垂直位移
    'api_break_x_arm',            # 水平位移（投手手臂方向）
    'api_break_x_batter_in',      # 水平位移（打者視角）

    # ── 出手點 ────────────────────────────
    'release_pos_x',        # 出手水平位置 (英尺)
    'release_pos_z',        # 出手垂直高度 (英尺)
    'release_pos_y',        # 出手縱深 (英尺)
    'release_extension',    # 出手延伸距離 (英尺)
    'arm_angle',            # 手臂角度 (度)

    # ── 球抵達本壘板的位置 ────────────────
    'plate_x',              # 水平位置 (英尺)
    'plate_z',              # 垂直高度 (英尺)
    'zone',                 # 好球帶區域 (1-14)

    # ── 初速與加速度（飛行物理）──────────
    'vx0', 'vy0', 'vz0',   # 出手初速三分量
    'ax', 'ay', 'az',       # 加速度三分量（含空氣阻力與馬格努斯力）

    # ── 好球帶邊界（打者身高相關）────────
    'sz_top',               # 好球帶上界 (英尺)
    'sz_bot',               # 好球帶下界 (英尺)

    # ── 投手手性（情境特徵）──────────────
    'p_throws',             # 投手慣用手 L / R
    'stand',                # 打者站位 L / R
    # ── 投手名稱 ──────────────
    'pitcher',
]


#%%
# ============================================================
# STEP 1：載入資料
# ============================================================
print("載入 CSV 中...")
df_raw = pd.read_csv('statcast_bat_tracking_2024_2025.csv',
                     usecols=lambda c: c in PHYSICAL_COLS)
print(f"原始資料：{len(df_raw):,} 筆，{df_raw.shape[1]} 欄")


#%%
# ============================================================
# STEP 2：基本清理
# ============================================================
df_raw = df_raw.dropna(subset=['pitch_type'])

name_map = (
    df_raw.dropna(subset=['pitch_type', 'pitch_name'])
    .drop_duplicates(subset='pitch_type')[['pitch_type', 'pitch_name']]
    .set_index('pitch_type')['pitch_name']
)
counts = df_raw['pitch_type'].value_counts()
summary = pd.DataFrame({
    'pitch_type': counts.index,
    'pitch_name': counts.index.map(name_map),
    'count':      counts.values,
    'ratio(%)':   (counts.values / len(df_raw) * 100).round(2)
}).reset_index(drop=True)
summary.index += 1
print("\n===== 所有球種統計 =====")
print(summary.to_string())

min_ratio = 0.01
total = len(df_raw)
valid_types = counts[counts / total >= min_ratio].index
df_raw = df_raw[df_raw['pitch_type'].isin(valid_types)]
print(f"\n保留球種（佔比 >= 1%）：")
print(df_raw['pitch_name'].value_counts().to_string())

num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
missing = df_raw[num_cols].isnull().mean().sort_values(ascending=False)
print(f"\n缺值比率（前 10）：")
print(missing.head(10).round(3).to_string())


def remove_outliers_iqr_by_group(df, cols, group_col='pitch_type', k=3.0):
    def get_group_mask(group):
        mask = pd.Series(True, index=group.index)
        for col in cols:
            # 如果該欄位在該球種內全是 NaN，則跳過
            if group[col].isnull().all():
                continue
            Q1 = group[col].quantile(0.25)
            Q3 = group[col].quantile(0.75)
            IQR = Q3 - Q1
            # 建立該組內的過濾條件
            mask &= group[col].between(Q1 - k * IQR, Q3 + k * IQR)
        return group[mask]

    # 利用 groupby().apply() 對每個球種分別清洗，再合併回來
    return df.groupby(group_col, group_keys=False).apply(get_group_mask)

n_before = len(df_raw)
# 傳入 'pitch_type' 作為分組依據
df_raw = remove_outliers_iqr_by_group(df_raw, num_cols, group_col='pitch_type', k=3.0)

print(f"\n離群值移除（分組 IQR）：{n_before - len(df_raw):,} 筆")
print(f"剩餘資料：{len(df_raw):,} 筆")


#%%合併KC&CU
df_raw['pitch_type'] = df_raw['pitch_type'].replace('KC', 'CU')

#%%

# ============================================================
# STEP 2.5：鏡像處理 — 消除左右投造成的雙峰
# ============================================================
print("\n===== STEP 2.5：鏡像處理 =====")
print(f"左投筆數：{(df_raw['p_throws']=='L').sum():,}")
print(f"右投筆數：{(df_raw['p_throws']=='R').sum():,}")

df_no_mirror = df_raw.copy()
df_no_mirror.to_csv('Pitch_physical_only.csv', index=False)
print(f"\n已儲存：Pitch_physical_only.csv（無鏡像，{len(df_no_mirror):,} 筆）")

df_mirror = df_raw.copy()
mask_L    = df_mirror['p_throws'] == 'L'

for col in ['pfx_x', 'ax', 'vx0']:
    df_mirror.loc[mask_L, col] = -df_mirror.loc[mask_L, col]
    print(f"  {col}：左投取負號")

def circular_mean(angles_deg):
    rad = np.deg2rad(angles_deg.dropna())
    return np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360

m_R = circular_mean(df_mirror[df_mirror['p_throws'] == 'R']['spin_axis'])
m_L = circular_mean(df_mirror[df_mirror['p_throws'] == 'L']['spin_axis'])
MIRROR_AXIS = ((m_R + m_L) / 2) % 360
df_mirror.loc[mask_L, 'spin_axis'] = (
    2 * MIRROR_AXIS - df_mirror.loc[mask_L, 'spin_axis']
) % 360
print(f"  spin_axis：右投均值 {m_R:.1f}°，左投均值 {m_L:.1f}°，鏡像軸 {MIRROR_AXIS:.1f}°")

print("\n  驗證（均值差越小代表雙峰消除越好）：")
print(f"  {'特徵':<10} {'鏡像前差距':>12} {'鏡像後差距':>12}")
print(f"  {'-'*36}")
for feat in ['pfx_x', 'ax', 'vx0']:
    before = abs(df_raw[df_raw['p_throws']=='R'][feat].mean() -
                 df_raw[df_raw['p_throws']=='L'][feat].mean())
    after  = abs(df_mirror[df_mirror['p_throws']=='R'][feat].mean() -
                 df_mirror[df_mirror['p_throws']=='L'][feat].mean())
    print(f"  {feat:<10} {before:>12.3f} {after:>12.3f}")

m_R_after = circular_mean(df_mirror[df_mirror['p_throws']=='R']['spin_axis'])
m_L_after = circular_mean(df_mirror[df_mirror['p_throws']=='L']['spin_axis'])
print(f"  {'spin_axis':<10} 鏡像前差 {abs(m_R-m_L):.1f}°  鏡像後差 {abs(m_R_after-m_L_after):.1f}°")

df_mirror.to_csv('Pitch_physical_only_mirror.csv', index=False)
print(f"\n已儲存：Pitch_physical_only_mirror.csv（有鏡像，{len(df_mirror):,} 筆）")

df_raw = df_mirror

#%%
# ============================================================
# STEP 3：取 10 萬筆（依球種分層抽樣）
# ============================================================
N_SAMPLE = 100_000

def stratified_sample(data, n):
    return (
        data
        .groupby('pitch_type', group_keys=False)
        .apply(lambda g: g.sample(
            n=min(len(g), int(n * len(g) / len(data))),
            random_state=42
        ))
        .reset_index(drop=True)
    )

df_sample_no_mirror = stratified_sample(df_no_mirror, N_SAMPLE)
df_sample_no_mirror.to_csv('testdata_only_phy.csv', index=False)
print(f"已儲存：testdata_only_phy.csv（無鏡像，{len(df_sample_no_mirror):,} 筆）")
print(df_sample_no_mirror['pitch_name'].value_counts().to_string())

df_sample_mirror = stratified_sample(df_mirror, N_SAMPLE)
df_sample_mirror.to_csv('testdata_only_phy_mirror.csv', index=False)
print(f"\n已儲存：testdata_only_phy_mirror.csv（有鏡像，{len(df_sample_mirror):,} 筆）")
print(df_sample_mirror['pitch_name'].value_counts().to_string())


#%%
# ============================================================
# STEP 4：EDA
# ============================================================
df_core_raw = pd.read_csv('testdata_only_phy_mirror.csv')

EXCLUDE = ['pitch_type', 'pitch_name', 'p_throws', 'stand']
ALL_NUMERIC = [
    col for col in df_core_raw.columns
    if col not in EXCLUDE
    and df_core_raw[col].dtype in [np.float64, np.int64]
    and df_core_raw[col].isnull().mean() < 0.3
]
print(f"進入分析的特徵數：{len(ALL_NUMERIC)} 個")
print(ALL_NUMERIC)

df_core = df_core_raw[ALL_NUMERIC + ['pitch_name']].dropna()
pitch_names = sorted(df_core['pitch_name'].unique())

print("\n===== 各球種特徵平均值 =====")
print(df_core.groupby('pitch_name')[ALL_NUMERIC].mean().round(2).to_string())

df_ana = df_core_raw[ALL_NUMERIC + ['pitch_name']].dropna(subset=['pitch_name'])
pitch_names_ana = sorted(df_ana['pitch_name'].dropna().unique())
n_total  = len(df_ana)
n_groups = len(pitch_names_ana)

print("\n===== 所有物理特徵鑑別力（F-ratio）=====")
f_ratios = {}
for feat in ALL_NUMERIC:
    groups = [df_ana[df_ana['pitch_name'] == p][feat].dropna()
              for p in pitch_names_ana]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        continue
    grand_mean = df_ana[feat].mean()
    between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups) / (n_groups - 1)
    within  = sum((len(g) - 1) * g.var() for g in groups) / (n_total - n_groups)
    f_ratios[feat] = round(between / within, 1) if within > 0 else 0

f_series = pd.Series(f_ratios).sort_values(ascending=False)
max_f = f_series.max()
for feat, f in f_series.items():
    bar = '█' * int(f / max_f * 35)
    missing_pct = df_core_raw[feat].isnull().mean() * 100
    print(f"  {feat:<35}  F={f:>8.1f}  缺值{missing_pct:4.1f}%  {bar}")

#%%
# ============================================================
# STEP 5：相關係數分析 — 找出重複特徵
# ============================================================
print("\n===== STEP 5：特徵相關係數分析 =====")

df_corr = df_core_raw[ALL_NUMERIC].dropna()
corr_matrix = df_corr.corr()

fig, ax = plt.subplots(figsize=(len(ALL_NUMERIC) * 0.9 + 2,
                                len(ALL_NUMERIC) * 0.9 + 2))

im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, shrink=0.8)

ax.set_xticks(range(len(ALL_NUMERIC)))
ax.set_yticks(range(len(ALL_NUMERIC)))
ax.set_xticklabels(ALL_NUMERIC, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ALL_NUMERIC, fontsize=8)

for i in range(len(ALL_NUMERIC)):
    for j in range(len(ALL_NUMERIC)):
        val = corr_matrix.values[i, j]
        if abs(val) > 0.5 and i != j:
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6.5,
                    color='white' if abs(val) > 0.75 else 'black')

ax.set_title('特徵相關係數矩陣（鏡像處理後）', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("已儲存：correlation_heatmap.png")

CORR_THRESHOLD = 0.9

print(f"\n高度相關特徵對（|r| > {CORR_THRESHOLD}）：")
print(f"{'特徵 A':<30} {'特徵 B':<30} {'相關係數':>8}  建議")
print("-" * 80)

high_corr_pairs = []

for i, feat_a in enumerate(ALL_NUMERIC):
    for j, feat_b in enumerate(ALL_NUMERIC):
        if j <= i:
            continue
        r = corr_matrix.loc[feat_a, feat_b]
        if abs(r) >= CORR_THRESHOLD:
            keep    = feat_a if f_ratios.get(feat_a, 0) >= f_ratios.get(feat_b, 0) else feat_b
            discard = feat_b if keep == feat_a else feat_a
            high_corr_pairs.append((feat_a, feat_b, r, keep, discard))
            print(f"{feat_a:<30} {feat_b:<30} {r:>8.3f}  保留 {keep}，移除 {discard}")

to_remove = set()
for _, _, _, keep, discard in high_corr_pairs:
    to_remove.add(discard)

FINAL_FEATURES = [f for f in ALL_NUMERIC if f not in to_remove]

print(f"\n移除高度重複特徵（{len(to_remove)} 個）：{sorted(to_remove)}")
print(f"\n建議最終特徵清單（{len(FINAL_FEATURES)} 個）：")
print(f"{'特徵':<30} {'F-ratio':>8}")
print("-" * 42)
for feat in sorted(FINAL_FEATURES, key=lambda x: f_ratios.get(x, 0), reverse=True):
    print(f"{feat:<30} {f_ratios.get(feat, 0):>8.1f}")

print(f"""
結論說明：
  相關係數 |r| > {CORR_THRESHOLD} → 兩個特徵幾乎攜帶相同資訊
  保留原則：F-ratio 較高的特徵（對球種分類貢獻較大）
  最終保留 {len(FINAL_FEATURES)} 個特徵供後續分類模型使用
""")


# %%
# ============================================================
# STEP 6：左右投子集球種分布診斷
# ============================================================
print("\n===== STEP 6：左右投子集球種分布診斷 =====")

df_full = pd.read_csv('testdata_only_phy.csv')

df_R = df_full[df_full['p_throws'] == 'R'].reset_index(drop=True)
df_L = df_full[df_full['p_throws'] == 'L'].reset_index(drop=True)

MIN_SAMPLES = 200

for label, df_sub in [('右投 (R)', df_R), ('左投 (L)', df_L)]:
    print(f"\n{'='*50}")
    print(f"  {label}：{len(df_sub):,} 筆")
    print(f"{'='*50}")
    print(f"  {'球種':<8} {'筆數':>8}   {'佔比':>6}   狀態")
    print(f"  {'-'*40}")

    counts_sub = df_sub['pitch_type'].value_counts()
    total_sub  = len(df_sub)

    for pitch, n in counts_sub.items():
        pct  = n / total_sub * 100
        flag = '✅' if n >= MIN_SAMPLES else '⚠  樣本不足'
        print(f"  {pitch:<8} {n:>8,}   {pct:>5.1f}%   {flag}")

    insufficient = counts_sub[counts_sub < MIN_SAMPLES]
    if len(insufficient) > 0:
        print(f"\n  ⚠  樣本不足球種：", list(insufficient.index))
    else:
        print(f"\n  ✅ 所有球種均 >= {MIN_SAMPLES} 筆")
# %%
