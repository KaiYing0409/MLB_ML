"""
MLB Statcast 投球資料 — EDA
============================
敘事目標：
  Part 1 → 看球種統計，決定分析哪些球種（佔比 >= 1%）
  Part 2 → 看所有物理特徵的左右投分布差異，
            發現雙峰現象 → 支持「左右投應分開處理」的決策
  Part 3 → 深入看 spin_axis，
            發現循環邊界問題 → 支持後續 sin/cos 轉換的決策

輸入：statcast_bat_tracking_2024_2025.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Noto Serif TC'
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = 'statcast_bat_tracking_2024_2025.csv'
MIN_RATIO = 0.01   # 佔比門檻

# 分析用物理特徵（排除 zone, sz_top, sz_bot, pitcher, stand 等非物理欄位）
PHYSICAL_FEATURES = [
    'release_speed', 'effective_speed',
    'release_spin_rate', 'spin_axis',
    'pfx_x', 'pfx_z', 'api_break_z_with_gravity',
    'api_break_x_arm', 'api_break_x_batter_in',
    'release_pos_x', 'release_pos_z', 'release_pos_y',
    'release_extension', 'arm_angle',
    'vx0', 'vy0', 'vz0',
    'ax', 'ay', 'az',
]

PITCH_ORDER = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS']

PITCH_COLORS = {
    'FF': '#1565C0', 'SI': '#42A5F5', 'FC': '#90CAF9',
    'SL': '#E53935', 'ST': '#FF8A65', 'CU': '#B71C1C',
    'CH': '#2E7D32', 'FS': '#A5D6A7',
}

# ============================================================
# 載入
# ============================================================
print("載入資料中...")
load_cols = ['pitch_type', 'p_throws'] + PHYSICAL_FEATURES
df = pd.read_csv(DATA_PATH, usecols=lambda c: c in load_cols)
df = df.dropna(subset=['pitch_type', 'p_throws'])
df['pitch_type'] = df['pitch_type'].replace('KC', 'CU')
print(f"載入完成：{len(df):,} 筆")

# ============================================================
# Part 1：球種統計 & 佔比篩選
# ============================================================
print(f"\n{'='*55}")
print("  Part 1：球種統計")
print(f"{'='*55}")

counts = df['pitch_type'].value_counts()
total  = len(df)

print(f"  {'球種':<8} {'筆數':>10} {'佔比':>8}   狀態")
print(f"  {'-'*40}")
for pt, n in counts.items():
    ratio = n / total
    flag  = '✅ 納入分析' if ratio >= MIN_RATIO else '— 排除'
    print(f"  {pt:<8} {n:>10,} {ratio:>8.2%}   {flag}")

valid_types = counts[counts / total >= MIN_RATIO].index.tolist()
df = df[df['pitch_type'].isin(valid_types)].copy()
PITCH_ORDER = [p for p in PITCH_ORDER if p in valid_types]

print(f"\n  → 保留 {len(PITCH_ORDER)} 種球種（佔比 >= {MIN_RATIO:.0%}）：{PITCH_ORDER}")
print(f"  → 後續分析資料：{len(df):,} 筆")

# ============================================================
# Part 2：物理特徵左右投分布差異——點圖
# ============================================================
print(f"\n{'='*55}")
print("  Part 2：物理特徵左右投分布差異")
print(f"{'='*55}")

# 排除 spin_axis（Part 3 單獨處理）
plot_features = [f for f in PHYSICAL_FEATURES
                 if f in df.columns and f != 'spin_axis']

N_COLS = 4
N_ROWS = int(np.ceil(len(plot_features) / N_COLS))

fig, axes = plt.subplots(N_ROWS, N_COLS,
                         figsize=(N_COLS * 5, N_ROWS * 4))
axes = axes.flatten()

for ax_idx, feat in enumerate(plot_features):
    ax = axes[ax_idx]

    for i, pt in enumerate(PITCH_ORDER):
        color = PITCH_COLORS.get(pt, '#888888')

        for hand, marker, ls, offset in [('R', 'o', '-',  0.15),
                                          ('L', 's', '--', -0.15)]:
            sub = df[(df['pitch_type'] == pt) &
                     (df['p_throws'] == hand)][feat].dropna()
            if len(sub) == 0:
                continue
            q1  = sub.quantile(0.25)
            med = sub.median()
            q3  = sub.quantile(0.75)
            y   = i + offset

            ax.plot([q1, q3], [y, y],
                    color=color, linewidth=1.8,
                    linestyle=ls, alpha=0.85)
            ax.plot(med, y,
                    marker=marker, color=color,
                    markersize=6, zorder=3)

    ax.set_yticks(np.arange(len(PITCH_ORDER)))
    ax.set_yticklabels(PITCH_ORDER, fontsize=8)
    ax.axvline(0, color='#AAAAAA', linewidth=0.8, linestyle=':')
    ax.set_title(feat, fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', labelsize=7)
    ax.invert_yaxis()

for ax_idx in range(len(plot_features), len(axes)):
    axes[ax_idx].set_visible(False)

r_line = plt.Line2D([0], [0], color='gray', linewidth=2,
                    linestyle='-', marker='o', markersize=6,
                    label='右投 R（實線・圓點）')
l_line = plt.Line2D([0], [0], color='gray', linewidth=2,
                    linestyle='--', marker='s', markersize=6,
                    label='左投 L（虛線・方點）')
fig.legend(handles=[r_line, l_line],
           loc='lower center', ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle('各物理特徵左右投分布差異（中位數 + Q1–Q3 範圍）',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('eda_feature_dotplot.png', dpi=130, bbox_inches='tight')
plt.show()
print("  已儲存：eda_feature_dotplot.png")

# ============================================================
# Part 3：spin_axis 專項分析——8×2 histogram
# ============================================================
print(f"\n{'='*55}")
print("  Part 3：spin_axis 左右投分布（循環邊界分析）")
print(f"{'='*55}")

HAND_COLORS = {'R': '#2196F3', 'L': '#F44336'}

fig, axes = plt.subplots(len(PITCH_ORDER), 2,
                         figsize=(12, len(PITCH_ORDER) * 3.2))
fig.suptitle('spin_axis 分布（左欄=右投 R，右欄=左投 L）\n觀察是否存在 0°/360° 循環邊界問題',
             fontsize=13, fontweight='bold', y=1.01)

for row, pt in enumerate(PITCH_ORDER):
    for col, hand in enumerate(['R', 'L']):
        ax  = axes[row, col]
        sub = df[(df['pitch_type'] == pt) &
                 (df['p_throws'] == hand)]['spin_axis'].dropna()

        if len(sub) == 0:
            ax.text(0.5, 0.5, '無資料', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{pt} — {hand}投', fontsize=9)
            continue

        ax.hist(sub, bins=60, range=(0, 360),
                color=HAND_COLORS[hand], alpha=0.75, edgecolor='none')
        ax.axvline(sub.median(), color='black', linewidth=1.5,
                   linestyle='--', label=f'median={sub.median():.0f}°')

        # 標示邊界區域
        ax.axvspan(0,   20,  alpha=0.08, color='red')
        ax.axvspan(340, 360, alpha=0.08, color='red')

        n_boundary = ((sub < 20) | (sub > 340)).sum()
        ax.set_title(
            f'{pt} — {"右" if hand=="R" else "左"}投  '
            f'(N={len(sub):,}，邊界樣本={n_boundary})',
            fontsize=8.5
        )
        ax.set_xlim(0, 360)
        ax.set_xlabel('spin_axis (°)', fontsize=7)
        ax.set_ylabel('count', fontsize=7)
        ax.legend(fontsize=6.5, loc='upper right')
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('eda_spin_axis.png', dpi=130, bbox_inches='tight')
plt.show()
print("  已儲存：eda_spin_axis.png")

# 邊界樣本統計摘要
print(f"\n  spin_axis 邊界樣本摘要（< 20° 或 > 340°）：")
print(f"  {'球種':<6} {'手性':<5} {'總數':>8} {'邊界樣本':>10} {'比例':>8}")
print(f"  {'-'*42}")
for pt in PITCH_ORDER:
    for hand in ['R', 'L']:
        sub = df[(df['pitch_type'] == pt) &
                 (df['p_throws'] == hand)]['spin_axis'].dropna()
        if len(sub) == 0:
            continue
        n_b = ((sub < 20) | (sub > 340)).sum()
        print(f"  {pt:<6} {hand:<5} {len(sub):>8,} {n_b:>10,} {n_b/len(sub):>8.2%}")
