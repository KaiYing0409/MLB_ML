"""
Step 3 — 特徵選擇
==================
輸入：testdata_only_phy.csv
      （已含 spin_axis_sin, spin_axis_cos，由 step2_preprocess.py 產出）

流程：
  1. 全體相關係數篩選（合併左右投，|r| > 0.9 去除冗餘特徵）
  2. 全體 F-ratio（三大類標籤，左右投分開）→ 特徵排序
  3. 各大類內部 F-ratio（對球種標籤，左右投分開）→ 特徵排序

輸出：
  - 印出各大類的特徵排名清單（供分類器腳本使用）
  - step3_group_fratio.png（全體 F-ratio 長條圖）

不包含：
  - 分類器訓練
  - Sequential Forward Selection（需要分類器，放在分類器腳本）
  - TOP_N 決定
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Noto Serif TC'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 設定
# ============================================================

INPUT_PATH  = 'testdata_only_phy.csv'
CORR_THRESH = 0.9

GROUP_MAP = {
    'FF': 'Fastball', 'SI': 'Fastball', 'FC': 'Fastball',
    'SL': 'Breaking', 'ST': 'Breaking', 'CU': 'Breaking',
    'CH': 'Offspeed', 'FS': 'Offspeed',
}
GROUPS = ['Fastball', 'Breaking', 'Offspeed']

PITCH_IN_GROUP = {
    'Fastball': ['FF', 'SI', 'FC'],
    'Breaking': ['SL', 'ST', 'CU'],
    'Offspeed': ['CH', 'FS'],
}

# spin_axis 已轉換為 sin/cos，不使用原始角度值
ALL_FEATS = [
    'release_speed', 'effective_speed',
    'release_spin_rate',
    'spin_axis_sin', 'spin_axis_cos',
    'pfx_x', 'pfx_z',
    'api_break_z_with_gravity',
    'vx0', 'vy0', 'vz0',
    'ax', 'ay', 'az',
]

# ============================================================
# 工具函式
# ============================================================

def zscore(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1e-9)

def compute_f_ratio(x_scaled, y, groups):
    """單一特徵對多類別的 F-ratio"""
    N = len(x_scaled)
    K = len(groups)
    grand_mean = x_scaled.mean()
    SS_b, SS_w = 0.0, 0.0
    for g in groups:
        Xg = x_scaled[y == g]
        if len(Xg) == 0:
            continue
        cm    = Xg.mean()
        SS_b += len(Xg) * (cm - grand_mean) ** 2
        SS_w += ((Xg - cm) ** 2).sum()
    MS_b = SS_b / (K - 1)
    MS_w = SS_w / (N - K) + 1e-9
    return MS_b / MS_w

def corr_filter(df_data, feat_list, fr_dict, thresh):
    """
    相關係數篩選：|r| > thresh 的特徵對，移除 F-ratio 較低的。
    fr_dict: {feature: F-ratio}
    回傳 (kept_list, log_list)
    """
    X     = df_data[feat_list].values.astype(float)
    mu    = X.mean(axis=0)
    sig   = X.std(axis=0) + 1e-9
    X_sc  = (X - mu) / sig
    corr  = np.corrcoef(X_sc.T)

    removed = set()
    log     = []
    for i in range(len(feat_list)):
        if feat_list[i] in removed:
            continue
        for j in range(i + 1, len(feat_list)):
            if feat_list[j] in removed:
                continue
            r = corr[i, j]
            if abs(r) >= thresh:
                fi      = fr_dict.get(feat_list[i], 0)
                fj      = fr_dict.get(feat_list[j], 0)
                keep    = feat_list[i] if fi >= fj else feat_list[j]
                discard = feat_list[j] if fi >= fj else feat_list[i]
                removed.add(discard)
                log.append((feat_list[i], feat_list[j], r, keep, discard))

    kept = [f for f in feat_list if f not in removed]
    return kept, log

# ============================================================
# 載入資料
# ============================================================

print("載入資料...")
df = pd.read_csv(INPUT_PATH)
df['group'] = df['pitch_type'].map(GROUP_MAP)
df = df.dropna(subset=['group']).reset_index(drop=True)

avail = [f for f in ALL_FEATS if f in df.columns]
df    = df.dropna(subset=avail).reset_index(drop=True)

df_R = df[df['p_throws'] == 'R'].reset_index(drop=True)
df_L = df[df['p_throws'] == 'L'].reset_index(drop=True)
print(f"右投：{len(df_R):,} 筆，左投：{len(df_L):,} 筆")

# ============================================================
# STEP 1：全體相關係數篩選（合併左右投）
# ============================================================

print(f"\n{'='*65}")
print(f"  STEP 1：相關係數篩選（左右投分開，取聯集，|r| > {CORR_THRESH}）")
print(f"{'='*65}")

# 左右投分開算 F-ratio，作為相關係數篩選的保留依據
fr_R_global = {feat: compute_f_ratio(zscore(df_R[feat].values), df_R['group'].values, GROUPS) for feat in avail}
fr_L_global = {feat: compute_f_ratio(zscore(df_L[feat].values), df_L['group'].values, GROUPS) for feat in avail}

# 左右投分開跑相關係數篩選，取移除特徵的聯集
kept_R, log_R = corr_filter(df_R, avail, fr_R_global, CORR_THRESH)
kept_L, log_L = corr_filter(df_L, avail, fr_L_global, CORR_THRESH)

removed_R   = set(d for _, _, _, _, d in log_R)
removed_L   = set(d for _, _, _, _, d in log_L)
removed_all = removed_R | removed_L  # 聯集：任一手性下重複就移除

for hand_label, log in [('右投 R', log_R), ('左投 L', log_L)]:
    print(f"\n  ── {hand_label} ──")
    if log:
        print(f"  {'特徵A':<26} {'特徵B':<26} {'r':>7}  {'保留':<24} 移除")
        print(f"  {'-'*95}")
        for fA, fB, r, keep, discard in log:
            print(f"  {fA:<26} {fB:<26} {r:>7.3f}  {keep:<24} {discard}")
    else:
        print("  （無高相關特徵對）")

kept = [f for f in avail if f not in removed_all]
print(f"\n  移除（{len(removed_all)} 個）：{sorted(removed_all)}")
print(f"  保留（{len(kept)} 個）：{kept}")

filtered_feats = kept

# ============================================================
# STEP 2：全體 F-ratio（三大類，左右投分開）
# ============================================================

print(f"\n{'='*65}")
print(f"  STEP 2：全體 F-ratio（三大類，左右投分開）")
print(f"{'='*65}")

global_results = []
for feat in filtered_feats:
    fr_R = compute_f_ratio(
        zscore(df_R[feat].values), df_R['group'].values, GROUPS
    )
    fr_L = compute_f_ratio(
        zscore(df_L[feat].values), df_L['group'].values, GROUPS
    )
    diff = abs(fr_R - fr_L) / (max(fr_R, fr_L) + 1e-9)
    global_results.append({'feature': feat, 'fr_R': fr_R, 'fr_L': fr_L, 'diff': diff})

global_df = (pd.DataFrame(global_results)
             .sort_values('fr_R', ascending=False)
             .reset_index(drop=True))

print(f"\n  {'排名':<5} {'特徵':<26} {'F-ratio R':>11} {'F-ratio L':>11} {'|R-L|/max':>11}")
print('  ' + '-' * 68)
for i, row in global_df.iterrows():
    flag = '  ⚠' if row['diff'] > 0.3 else ''
    print(f"  {i+1:<5} {row['feature']:<26} {row['fr_R']:>11.1f} {row['fr_L']:>11.1f} {row['diff']:>10.1%}{flag}")

# 繪圖
feats  = global_df['feature'].tolist()
fr_R_v = global_df['fr_R'].tolist()
fr_L_v = global_df['fr_L'].tolist()
n      = len(feats)
y      = np.arange(n)
height = 0.35

fig, ax = plt.subplots(figsize=(12, max(6, n * 0.5)))
bars_R = ax.barh(y + height/2, fr_R_v, height,
                 label='右投 R', color='#2166ac', alpha=0.85,
                 edgecolor='black', linewidth=0.5)
bars_L = ax.barh(y - height/2, fr_L_v, height,
                 label='左投 L', color='#d6604d', alpha=0.85,
                 edgecolor='black', linewidth=0.5)

max_val = max(fr_R_v + fr_L_v)
for bar, val in zip(bars_R, fr_R_v):
    ax.text(bar.get_width() + max_val * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.0f}', va='center', ha='left',
            fontsize=7.5, color='#2166ac')

ax.set_yticks(y)
ax.set_yticklabels(feats, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('F-ratio（三大類，z-score 標準化）', fontsize=10)
ax.set_title('Step 3 — 全體 F-ratio（三大類）：左右投分開\n依右投排序；⚠ = 左右投差異 > 30%',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='x', linewidth=0.5)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('step3_group_fratio.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  已儲存：step3_group_fratio.png")

# ============================================================
# STEP 3：各大類內部 F-ratio（對球種標籤，左右投分開）
# ============================================================

print(f"\n{'='*65}")
print(f"  STEP 3：各大類內部 F-ratio（對球種標籤）")
print(f"{'='*65}")

group_feat_results = {}

for grp, pitches in PITCH_IN_GROUP.items():
    print(f"\n{'='*55}")
    print(f"  大類：{grp}  （球種：{pitches}）")
    print(f"{'='*55}")

    mask_grp = df['pitch_type'].isin(pitches)
    df_grp_R = df[mask_grp & (df['p_throws'] == 'R')].reset_index(drop=True)
    df_grp_L = df[mask_grp & (df['p_throws'] == 'L')].reset_index(drop=True)
    print(f"  右投：{len(df_grp_R):,} 筆，左投：{len(df_grp_L):,} 筆")

    grp_results = []
    for feat in filtered_feats:
        fr_R = compute_f_ratio(
            zscore(df_grp_R[feat].values),
            df_grp_R['pitch_type'].values, pitches
        )
        fr_L = compute_f_ratio(
            zscore(df_grp_L[feat].values),
            df_grp_L['pitch_type'].values, pitches
        )
        diff = abs(fr_R - fr_L) / (max(fr_R, fr_L) + 1e-9)
        grp_results.append({'feature': feat, 'fr_R': fr_R, 'fr_L': fr_L, 'diff': diff})

    grp_df = (pd.DataFrame(grp_results)
              .sort_values('fr_R', ascending=False)
              .reset_index(drop=True))

    print(f"\n  {'排名':<5} {'特徵':<26} {'F R':>10} {'F L':>10} {'|R-L|/max':>11}")
    print('  ' + '-' * 65)
    for i, row in grp_df.iterrows():
        flag = '  ⚠' if row['diff'] > 0.3 else ''
        print(f"  {i+1:<5} {row['feature']:<26} {row['fr_R']:>10.1f} {row['fr_L']:>10.1f} {row['diff']:>10.1%}{flag}")

    group_feat_results[grp] = grp_df

# ============================================================
# 摘要
# ============================================================

print(f"\n{'='*65}")
print(f"  特徵選擇摘要")
print(f"{'='*65}")
print(f"\n  相關係數篩選後保留特徵（{len(filtered_feats)} 個）：")
print(f"  {filtered_feats}")

print(f"\n  各大類前 5 名特徵（依右投 F-ratio）：")
for grp, grp_df in group_feat_results.items():
    top5 = grp_df['feature'].head(5).tolist()
    print(f"  {grp:<12}：{top5}")

print(f"\n  ➜ TOP_N 由分類器腳本中的 Sequential Forward Selection 決定")