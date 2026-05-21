"""
棒球球種分類器 — 完整 Pipeline
================================
流程：
  1. LDA 判斷左右投（不需要使用者輸入 p_throws）
  2. 右投 → KNN_R 分類球種
     左投 → KNN_L 分類球種

特色：
  - 不做鏡像（避免 spin_axis 鏡像軸無法統一定義的問題）
  - 完全手刻，不使用 sklearn
  - 評分方式：各球種準確率加總平均（Macro Accuracy）

使用方式：
  python pitch_classifier_pipeline.py

輸出：
  - 各階段準確率
  - 最終球種分類 Macro Accuracy
  - Confusion Matrix
  - pipeline_results.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.family'] = 'Noto Serif TC'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 設定
# ============================================================

DATA_PATH    = 'testdata_only_phy.csv'   # 無鏡像版本
TEST_RATIO   = 0.2
K            = 10                         # KNN 的 k 值
MIN_SAMPLES  = 200                        # 球種最少樣本門檻
RANDOM_SEED  = 42

# KNN 使用的球種分類特徵
PITCH_FEATURES = [
    'api_break_x_arm',
    'api_break_z_with_gravity',
    'spin_axis',
    'release_speed',
    'release_spin_rate',
    'ay',
]

# LDA 使用的左右投特徵
LDA_FEATURES = [
    'pfx_x',
    'ax',
    'vx0',
    'spin_axis',
    'arm_angle',
    'api_break_x_arm',
    'release_speed',
    'release_spin_rate',
]

# ============================================================
# 工具函數
# ============================================================

def zscore(X_train, X_test):
    """用訓練集的 mean/std 標準化，避免 data leakage"""
    mu  = X_train.mean(axis=0)
    sig = X_train.std(axis=0) + 1e-9
    return (X_train - mu) / sig, (X_test - mu) / sig, mu, sig

def macro_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    accs = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        accs.append(np.mean(y_pred[mask] == c))
    return np.mean(accs)

def per_class_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    result = {}
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        result[c] = np.mean(y_pred[mask] == c)
    return result

# ============================================================
# LDA 二元分類器（手刻）
# ============================================================

class BinaryLDA:
    """Fisher's Linear Discriminant Analysis（二元分類）"""

    def fit(self, X, y, class_pos='R', class_neg='L'):
        self.class_pos = class_pos
        self.class_neg = class_neg

        X_pos = X[y == class_pos]
        X_neg = X[y == class_neg]

        m_pos = X_pos.mean(axis=0)
        m_neg = X_neg.mean(axis=0)

        S_pos = (X_pos - m_pos).T @ (X_pos - m_pos)
        S_neg = (X_neg - m_neg).T @ (X_neg - m_neg)
        S_W   = S_pos + S_neg + np.eye(X.shape[1]) * 1e-6

        self.w = np.linalg.solve(S_W, m_pos - m_neg)
        self.w /= np.linalg.norm(self.w)

        z_pos = X_pos @ self.w
        z_neg = X_neg @ self.w
        self.threshold = (z_pos.mean() + z_neg.mean()) / 2

        between = (z_pos.mean() - z_neg.mean()) ** 2
        within  = z_pos.var() + z_neg.var()
        self.J  = between / (within + 1e-9)

    def predict(self, X):
        z = X @ self.w
        return np.where(z >= self.threshold, self.class_pos, self.class_neg)


# ============================================================
# KNN 分類器（手刻）
# ============================================================

class KNNClassifier:
    """K-Nearest Neighbors 分類器（純 NumPy 實作）"""

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, batch_size=500):
        preds = []
        n = len(X)
        for start in range(0, n, batch_size):
            end   = min(start + batch_size, n)
            batch = X[start:end]
            diffs = batch[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            for i in range(len(batch)):
                nn_idx    = np.argpartition(dists[i], self.k)[:self.k]
                nn_labels = self.y_train[nn_idx]
                vote      = Counter(nn_labels)
                preds.append(vote.most_common(1)[0][0])
            print(f"  預測進度：{end}/{n} ({end/n*100:.1f}%)", end='\r')
        print()
        return np.array(preds)


# ============================================================
# STEP 1：讀資料 & 切分
# ============================================================

print("=" * 55)
print("  棒球球種分類器 Pipeline")
print("=" * 55)

print("\n[STEP 1] 讀資料")
df = pd.read_csv(DATA_PATH)

# 過濾球種樣本不足
counts = df['pitch_type'].value_counts()
valid  = counts[counts >= MIN_SAMPLES].index
df     = df[df['pitch_type'].isin(valid)].reset_index(drop=True)
df     = df[df['p_throws'].isin(['R', 'L'])].reset_index(drop=True)

# 確認所有特徵都存在
lda_feats   = [f for f in LDA_FEATURES   if f in df.columns]
pitch_feats = [f for f in PITCH_FEATURES if f in df.columns]
needed_cols = list(set(lda_feats + pitch_feats + ['pitch_type', 'p_throws']))
df = df[needed_cols].dropna().reset_index(drop=True)

print(f"  建模資料：{len(df):,} 筆，{df['pitch_type'].nunique()} 種球種")
print(f"  右投：{(df['p_throws']=='R').sum():,} 筆  左投：{(df['p_throws']=='L').sum():,} 筆")
print(f"  球種分布：")
for pt, n in df['pitch_type'].value_counts().items():
    print(f"    {pt:<6} {n:>6,} 筆")

# 全局 train/test split（確保左右投比例一致）
np.random.seed(RANDOM_SEED)
idx    = np.random.permutation(len(df))
n_test = int(len(df) * TEST_RATIO)
test_idx  = idx[:n_test]
train_idx = idx[n_test:]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_test  = df.iloc[test_idx].reset_index(drop=True)

print(f"\n  訓練集：{len(df_train):,} 筆  測試集：{len(df_test):,} 筆")

# ============================================================
# STEP 2：訓練 LDA 左右投分類器
# ============================================================

print("\n[STEP 2] 訓練 LDA 左右投分類器")

X_lda_train = df_train[lda_feats].values.astype(float)
y_lda_train = df_train['p_throws'].values
X_lda_test  = df_test[lda_feats].values.astype(float)
y_lda_test  = df_test['p_throws'].values

# 標準化
X_lda_train_z, X_lda_test_z, lda_mu, lda_sig = zscore(X_lda_train, X_lda_test)

lda = BinaryLDA()
lda.fit(X_lda_train_z, y_lda_train, class_pos='R', class_neg='L')

# 在測試集上評估 LDA
lda_pred_test = lda.predict(X_lda_test_z)
lda_acc = np.mean(lda_pred_test == y_lda_test)
lda_acc_R = np.mean(lda_pred_test[y_lda_test=='R'] == 'R')
lda_acc_L = np.mean(lda_pred_test[y_lda_test=='L'] == 'L')

print(f"  LDA 整體準確率：{lda_acc:.4f} ({lda_acc*100:.2f}%)")
print(f"  右投準確率：{lda_acc_R*100:.2f}%  左投準確率：{lda_acc_L*100:.2f}%")
print(f"  Fisher Criterion J(w)：{lda.J:.4f}")

print(f"\n  LDA 特徵權重（|w| 由大到小）：")
for feat, w in sorted(zip(lda_feats, lda.w), key=lambda x: -abs(x[1])):
    bar  = '█' * int(abs(w) * 20)
    sign = '+' if w >= 0 else '-'
    print(f"    {feat:<30} {sign}{abs(w):.4f}  {bar}")

# 對整個訓練集也做預測（供後續 KNN 訓練分組用）
X_lda_all_z = (df_train[lda_feats].values.astype(float) - lda_mu) / lda_sig
lda_pred_train = lda.predict(X_lda_all_z)

# ============================================================
# STEP 3：訓練 KNN_R 和 KNN_L
# ============================================================

print("\n[STEP 3] 訓練 KNN 球種分類器（左右投分開）")

def train_knn(df_sub, label):
    """從 df_sub 訓練一個 KNN，回傳 (knn, mu, sig)"""
    X = df_sub[pitch_feats].values.astype(float)
    y = df_sub['pitch_type'].values
    mu  = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-9
    X_z = (X - mu) / sig
    knn = KNNClassifier(k=K)
    knn.fit(X_z, y)
    pitch_counts = pd.Series(y).value_counts()
    print(f"  {label}：{len(df_sub):,} 筆，{pitch_counts.to_dict()}")
    return knn, mu, sig

# 用 LDA 預測結果分組（訓練集）
df_train_R = df_train[lda_pred_train == 'R'].reset_index(drop=True)
df_train_L = df_train[lda_pred_train == 'L'].reset_index(drop=True)

knn_R, mu_R, sig_R = train_knn(df_train_R, 'KNN_R（右投）')
knn_L, mu_L, sig_L = train_knn(df_train_L, 'KNN_L（左投）')

# ============================================================
# STEP 4：Pipeline 預測
# ============================================================

print(f"\n[STEP 4] Pipeline 預測（測試集 {len(df_test):,} 筆）")

# LDA 判斷左右投
X_test_lda_z = (df_test[lda_feats].values.astype(float) - lda_mu) / lda_sig
lda_pred = lda.predict(X_test_lda_z)

# 依 LDA 結果分組，分別用 KNN_R / KNN_L 預測
y_final = np.empty(len(df_test), dtype=object)

idx_R = np.where(lda_pred == 'R')[0]
idx_L = np.where(lda_pred == 'L')[0]

print(f"  LDA 判斷右投：{len(idx_R):,} 筆  左投：{len(idx_L):,} 筆")

if len(idx_R) > 0:
    print(f"  KNN_R 預測中...")
    X_R = df_test.iloc[idx_R][pitch_feats].values.astype(float)
    X_R_z = (X_R - mu_R) / sig_R
    y_final[idx_R] = knn_R.predict(X_R_z)

if len(idx_L) > 0:
    print(f"  KNN_L 預測中...")
    X_L = df_test.iloc[idx_L][pitch_feats].values.astype(float)
    X_L_z = (X_L - mu_L) / sig_L
    y_final[idx_L] = knn_L.predict(X_L_z)

y_true = df_test['pitch_type'].values

# ============================================================
# STEP 5：評估
# ============================================================

print(f"\n[STEP 5] 評估結果")

pca_result = per_class_accuracy(y_true, y_final)
macro      = macro_accuracy(y_true, y_final)

print(f"\n  {'球種':<12} {'測試筆數':>8} {'準確率':>8}")
print(f"  {'-'*32}")
for cls in sorted(pca_result):
    n_cls = (y_true == cls).sum()
    print(f"  {cls:<12} {n_cls:>8,} {pca_result[cls]:>7.1%}")
print(f"  {'-'*32}")
print(f"  {'Macro Accuracy':.<20} {macro:.4f}  ({macro:.1%})")

# ============================================================
# STEP 6：Confusion Matrix
# ============================================================

print(f"\n[STEP 6] Confusion Matrix")
cls_list   = sorted(np.unique(y_true))
n_cls      = len(cls_list)
cls_to_idx = {c: i for i, c in enumerate(cls_list)}

cm = np.zeros((n_cls, n_cls), dtype=int)
for t, p in zip(y_true, y_final):
    if p is not None:
        cm[cls_to_idx[t]][cls_to_idx[str(p)]] += 1

header = f"  {'':>6}" + "".join(f"{c:>6}" for c in cls_list)
print(header)
for i, cls in enumerate(cls_list):
    row = f"  {cls:>6}" + "".join(f"{cm[i][j]:>6}" for j in range(n_cls))
    print(row)

# ============================================================
# STEP 7：視覺化
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Pipeline（LDA→KNN_R/KNN_L）  Macro Acc = {macro:.1%}', fontsize=13)

# 左圖：各球種準確率
sorted_cls = sorted(pca_result, key=lambda c: pca_result[c], reverse=True)
accs   = [pca_result[c] for c in sorted_cls]
colors = ['steelblue' if a >= macro else 'tomato' for a in accs]

axes[0].bar(sorted_cls, accs, color=colors, edgecolor='white')
axes[0].axhline(macro, color='black', linestyle='--', linewidth=1.2,
                label=f'Macro = {macro:.1%}')
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel('準確率')
axes[0].set_title('各球種準確率')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=30)

# 右圖：Confusion Matrix heatmap
im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_xticks(range(n_cls))
axes[1].set_yticks(range(n_cls))
axes[1].set_xticklabels(cls_list, rotation=45, ha='right', fontsize=8)
axes[1].set_yticklabels(cls_list, fontsize=8)
axes[1].set_xlabel('預測球種')
axes[1].set_ylabel('實際球種')
axes[1].set_title('Confusion Matrix')

if n_cls <= 12:
    for i in range(n_cls):
        for j in range(n_cls):
            val = cm[i][j]
            if val > 0:
                color = 'white' if val > cm.max() * 0.5 else 'black'
                axes[1].text(j, i, str(val), ha='center', va='center',
                             fontsize=7, color=color)

plt.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.savefig('pipeline_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n圖表已儲存：pipeline_results.png")

# ============================================================
# STEP 8：摘要
# ============================================================

print("\n" + "=" * 55)
print("  Pipeline 摘要")
print("=" * 55)
print(f"  LDA 左右投準確率  ：{lda_acc*100:.2f}%")
print(f"  KNN k 值          ：{K}")
print(f"  右投訓練筆數      ：{len(df_train_R):,}")
print(f"  左投訓練筆數      ：{len(df_train_L):,}")
print(f"  最終 Macro Acc    ：{macro:.4f} ({macro:.1%})")
print("=" * 55)
print("\n設計說明：")
print("  不做鏡像，改以 LDA 自動判斷左右投")
print("  spin_axis 保持原始值，避免鏡像軸定義模糊的問題")
print("  左右投各自在獨立特徵空間內計算距離，點群更集中")
