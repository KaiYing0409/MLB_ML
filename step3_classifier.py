"""
棒球球種分類器 — QDA + 信心門檻 + 第二層 LDA
================================================
架構：
  第一層：LDA 左右投 → QDA 球種分類（輸出真正的後驗機率）
  信心評估：top1 - top2 的後驗機率差距（margin）
  第二層：當 margin 不夠大且前兩名是預設混淆對時，用專屬 BinaryLDA 重判

QDA 對比 NB：
  NB 假設特徵之間條件獨立（covariance 為對角矩陣）
  QDA 估計每個球種各自的完整 covariance matrix
  → 可以捕捉特徵之間的相關性（例如球速和轉速的相關）
  → 判別函數為二次曲線，更有彈性

後驗機率計算（L6 公式）：
  gi(x) = -1/2 * log|Si| - 1/2 * (x-mi)^T Si^{-1} (x-mi) + log P(Ci)
  P(Ci|x) ∝ exp(gi(x))

評估方法論：60/20/20 嚴格切割
  訓練集：訓練所有模型
  驗證集：選 margin threshold
  測試集：只跑一次

使用方式：
  python pitch_classifier_qda.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Noto Serif TC'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 設定
# ============================================================

DATA_PATH   = 'testdata_relative_phy.csv'
TEST_RATIO  = 0.2
VAL_RATIO   = 0.2
MIN_SAMPLES = 200
RANDOM_SEED = 42

PITCH_FEATURES = [
    # api_break_x_arm/z_with_gravity 在 SL/FC/ST 的 F-ratio 大
    'api_break_x_arm',
    'api_break_z_with_gravity',
    'spin_axis_sin',
    'spin_axis_cos',
    'release_speed',
    'release_spin_rate',
    'ay',
    'vy0',
]

LDA_FEATURES = [
    'pfx_x', 'ax', 'vx0', 'spin_axis_sin', 'spin_axis_cos',
    'arm_angle', 'api_break_x_arm',
    'release_speed', 'release_spin_rate',
]

# weighted=False → 幾何中點（保護大宗球種）
# weighted=True  → 加權閾值（保護小宗球種）
PREDEFINED_PAIRS = {
    ('CH', 'FS'): {'feats': ['release_spin_rate', 'api_break_x_arm', 'spin_axis_sin', 'api_break_z_with_gravity'], 'weighted': False},
    ('FC', 'SL'): {'feats': ['api_break_z_with_gravity', 'release_speed', 'spin_axis_cos', 'api_break_x_arm'], 'weighted': True},
    ('SL', 'ST'): {'feats': ['api_break_x_arm', 'release_speed', 'spin_axis_cos', 'release_spin_rate'], 'weighted': True},
    ('CU', 'ST'): {'feats': ['api_break_z_with_gravity', 'spin_axis_cos', 'release_speed', 'api_break_x_arm'], 'weighted': True},
}

MARGIN_LIST = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

# ============================================================
# 工具函數
# ============================================================

def zscore_fit(X):
    mu  = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-9
    return mu, sig

def zscore_transform(X, mu, sig):
    return (X - mu) / sig

def macro_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    return np.mean([
        np.mean(y_pred[y_true == c] == c)
        for c in classes if (y_true == c).sum() > 0
    ])

def per_class_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    return {c: np.mean(y_pred[y_true == c] == c)
            for c in classes if (y_true == c).sum() > 0}

# ============================================================
# QDA 分類器（手刻，L6 公式）
# ============================================================

class QDAClassifier:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.params_  = {}
        N = len(X)
        for c in self.classes_:
            Xc = X[y == c]
            Nc = len(Xc)
            mi = Xc.mean(axis=0)
            diff = Xc - mi
            Si   = (diff.T @ diff) / (Nc - 1)
            Si += np.eye(X.shape[1]) * 1e-4
            Si_inv   = np.linalg.inv(Si)
            log_det  = np.log(np.linalg.det(Si) + 1e-300)
            log_prior = np.log(Nc / N)
            self.params_[c] = {
                'mi':        mi,
                'Si_inv':    Si_inv,
                'log_det':   log_det,
                'log_prior': log_prior,
            }

    def discriminant(self, X):
        n = len(X)
        K = len(self.classes_)
        G = np.zeros((n, K))
        for j, c in enumerate(self.classes_):
            p    = self.params_[c]
            diff = X - p['mi']
            maha = np.sum(diff @ p['Si_inv'] * diff, axis=1)
            G[:, j] = (
                -0.5 * p['log_det']
                - 0.5 * maha
                + p['log_prior']
            )
        return G

    def posterior(self, X):
        G = self.discriminant(X)
        G_shift = G - G.max(axis=1, keepdims=True)
        exp_G   = np.exp(G_shift)
        return exp_G / exp_G.sum(axis=1, keepdims=True)

    def predict_with_margin(self, X):
        probs    = self.posterior(X)
        top2_idx = np.argsort(probs, axis=1)[:, -2:][:, ::-1]
        preds    = np.array([self.classes_[i] for i in top2_idx[:, 0]])
        top2s    = np.array([self.classes_[i] for i in top2_idx[:, 1]])
        margins  = probs[np.arange(len(X)), top2_idx[:, 0]] - \
                   probs[np.arange(len(X)), top2_idx[:, 1]]
        return preds, margins, top2s

    def predict(self, X):
        preds, _, _ = self.predict_with_margin(X)
        return preds

# ============================================================
# BinaryLDA（第二層）
# ============================================================

class BinaryLDA:
    def fit(self, X, y, class_pos, class_neg, weighted=True):
        self.class_pos = class_pos
        self.class_neg = class_neg
        X_pos = X[y == class_pos]
        X_neg = X[y == class_neg]
        m_pos = X_pos.mean(axis=0)
        m_neg = X_neg.mean(axis=0)
        S_W = ((X_pos - m_pos).T @ (X_pos - m_pos) +
               (X_neg - m_neg).T @ (X_neg - m_neg) +
               np.eye(X.shape[1]) * 1e-6)
        self.w = np.linalg.solve(S_W, m_pos - m_neg)
        self.w /= np.linalg.norm(self.w)
        z_pos = X_pos @ self.w
        z_neg = X_neg @ self.w
        N_pos = len(X_pos)
        N_neg = len(X_neg)
        if weighted:
            self.threshold = (N_pos * z_pos.mean() + N_neg * z_neg.mean()) / (N_pos + N_neg)
        else:
            self.threshold = (z_pos.mean() + z_neg.mean()) / 2
        between = (z_pos.mean() - z_neg.mean()) ** 2
        within  = z_pos.var() + z_neg.var()
        self.J  = between / (within + 1e-9)

    def predict(self, X):
        return np.where(X @ self.w >= self.threshold,
                        self.class_pos, self.class_neg)

# ============================================================
# 第二層套用
# ============================================================

def apply_layer2(df_eval, y_l1, margins, top2s, layer2_models, margin_th):
    y_final  = y_l1.copy()
    n_layer2 = 0
    for i in range(len(df_eval)):
        if margins[i] >= margin_th:
            continue
        pair = tuple(sorted([y_l1[i], top2s[i]]))
        if pair not in layer2_models:
            continue
        lda2, mu2, sig2, feats2 = layer2_models[pair]
        x = df_eval.iloc[i][feats2].values.astype(float).reshape(1, -1)
        y_final[i] = lda2.predict(zscore_transform(x, mu2, sig2))[0]
        n_layer2 += 1
    return y_final, n_layer2

# ============================================================
# STEP 1：讀資料 + 三分切割
# ============================================================

print("=" * 60)
print("  QDA + 信心門檻 + 第二層 LDA（60/20/20）")
print("=" * 60)

print("\n[STEP 1] 讀資料 + 三分切割")
df = pd.read_csv(DATA_PATH)

counts = df['pitch_type'].value_counts()
valid  = counts[counts >= MIN_SAMPLES].index
df     = df[df['pitch_type'].isin(valid)].reset_index(drop=True)
df     = df[df['p_throws'].isin(['R', 'L'])].reset_index(drop=True)

all_feats = set(LDA_FEATURES + PITCH_FEATURES)
for config in PREDEFINED_PAIRS.values():
    all_feats.update(config['feats'])
all_feats.update(['pitch_type', 'p_throws'])
avail_cols = [c for c in all_feats if c in df.columns]
df = df[avail_cols].dropna().reset_index(drop=True)

print(f"  建模資料：{len(df):,} 筆，{df['pitch_type'].nunique()} 種球種")
for pt, n in df['pitch_type'].value_counts().items():
    print(f"    {pt:<6} {n:>6,} 筆")

np.random.seed(RANDOM_SEED)
idx      = np.random.permutation(len(df))
n_test   = int(len(df) * TEST_RATIO)
n_val    = int(len(df) * VAL_RATIO)
df_train = df.iloc[idx[n_test + n_val:]].reset_index(drop=True)
df_val   = df.iloc[idx[n_test:n_test + n_val]].reset_index(drop=True)
df_test  = df.iloc[idx[:n_test]].reset_index(drop=True)
print(f"\n  訓練集：{len(df_train):,}  驗證集：{len(df_val):,}  測試集：{len(df_test):,}")

# ============================================================
# STEP 2：訓練 LDA 左右投
# ============================================================

print("\n[STEP 2] 訓練 LDA 左右投分類器")
lda_feats = [f for f in LDA_FEATURES if f in df.columns]
X_lda_tr  = df_train[lda_feats].values.astype(float)
y_lda_tr  = df_train['p_throws'].values
lda_mu, lda_sig = zscore_fit(X_lda_tr)

lda_hand = BinaryLDA()
lda_hand.fit(zscore_transform(X_lda_tr, lda_mu, lda_sig),
             y_lda_tr, class_pos='R', class_neg='L')

def get_lda_pred(df_sub):
    X = df_sub[lda_feats].values.astype(float)
    return lda_hand.predict(zscore_transform(X, lda_mu, lda_sig))

lda_pred_tr  = get_lda_pred(df_train)
lda_pred_val = get_lda_pred(df_val)
lda_pred_te  = get_lda_pred(df_test)

lda_acc_val = np.mean(lda_pred_val == df_val['p_throws'].values)
print(f"  驗證集 LDA 準確率：{lda_acc_val*100:.2f}%")

# ============================================================
# STEP 3：訓練 QDA（左右投分開）
# ============================================================

print("\n[STEP 3] 訓練 QDA 球種分類器")
pitch_feats = [f for f in PITCH_FEATURES if f in df.columns]

df_tr_R = df_train[lda_pred_tr == 'R'].reset_index(drop=True)
df_tr_L = df_train[lda_pred_tr == 'L'].reset_index(drop=True)

def train_qda(df_sub, label):
    X = df_sub[pitch_feats].values.astype(float)
    y = df_sub['pitch_type'].values
    mu, sig = zscore_fit(X)
    qda = QDAClassifier()
    qda.fit(zscore_transform(X, mu, sig), y)
    print(f"  {label}：{len(df_sub):,} 筆，{len(np.unique(y))} 種球種")
    return qda, mu, sig

qda_R, mu_R, sig_R = train_qda(df_tr_R, 'QDA_R（右投）')
qda_L, mu_L, sig_L = train_qda(df_tr_L, 'QDA_L（左投）')

# ============================================================
# STEP 4：訓練第二層 BinaryLDA
# ============================================================

print("\n[STEP 4] 訓練第二層 BinaryLDA")
layer2_models = {}
for pair, config in PREDEFINED_PAIRS.items():
    class_a, class_b = pair
    feats    = config['feats']
    weighted = config['weighted']
    avail_feats = [f for f in feats if f in df.columns]
    if len(avail_feats) < 2:
        continue
    mask = df_train['pitch_type'].isin([class_a, class_b])
    df_pair = df_train[mask].reset_index(drop=True)
    X = df_pair[avail_feats].values.astype(float)
    y = df_pair['pitch_type'].values
    mu, sig = zscore_fit(X)
    lda2 = BinaryLDA()
    lda2.fit(zscore_transform(X, mu, sig), y,
             class_pos=class_a, class_neg=class_b, weighted=weighted)
    layer2_models[pair] = (lda2, mu, sig, avail_feats)
    print(f"  {class_a} vs {class_b}：J(w)={lda2.J:.2f}")


for pair, (lda2, mu, sig, feats) in layer2_models.items():
    print(f"  {pair}：threshold={lda2.threshold:.4f}")

# ============================================================
# STEP 5：QDA 預測驗證集
# ============================================================

print("\n[STEP 5] QDA 預測驗證集")

def qda_predict_full(df_eval, lda_pred):
    n = len(df_eval)
    y_l1    = np.empty(n, dtype=object)
    margins = np.empty(n, dtype=float)
    top2s   = np.empty(n, dtype=object)

    idx_R = np.where(lda_pred == 'R')[0]
    idx_L = np.where(lda_pred == 'L')[0]

    if len(idx_R) > 0:
        X_R = df_eval.iloc[idx_R][pitch_feats].values.astype(float)
        p, m, t = qda_R.predict_with_margin(zscore_transform(X_R, mu_R, sig_R))
        y_l1[idx_R], margins[idx_R], top2s[idx_R] = p, m, t

    if len(idx_L) > 0:
        X_L = df_eval.iloc[idx_L][pitch_feats].values.astype(float)
        p, m, t = qda_L.predict_with_margin(zscore_transform(X_L, mu_L, sig_L))
        y_l1[idx_L], margins[idx_L], top2s[idx_L] = p, m, t

    return y_l1, margins, top2s

y_val_l1, val_margins, val_top2s = qda_predict_full(df_val, lda_pred_val)
y_val_true = df_val['pitch_type'].values
macro_l1_val = macro_accuracy(y_val_true, y_val_l1)
print(f"  QDA 第一層驗證集 Macro：{macro_l1_val:.4f}")

print(f"\n  margin 分布：")
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    n_below = (val_margins < threshold).sum()
    print(f"    margin < {threshold:.2f}：{n_below:,} 筆 ({n_below/len(df_val)*100:.1f}%)")

# ============================================================
# STEP 6：驗證集掃描 margin threshold
# ============================================================

print(f"\n[STEP 6] 驗證集掃描 margin threshold")

results    = []
best_macro = 0
best_th    = None

print(f"  {'threshold':>10} {'L1 Macro':>10} {'L2 Macro':>10} {'第二層觸發':>10}")
print(f"  {'-'*46}")

for th in MARGIN_LIST:
    y_l2, n_l2 = apply_layer2(df_val, y_val_l1, val_margins,
                               val_top2s, layer2_models, th)
    m2 = macro_accuracy(y_val_true, y_l2)
    results.append((th, macro_l1_val, m2, n_l2))

    marker = ''
    if m2 > best_macro:
        best_macro = m2
        best_th    = th
        marker = ' ← 最佳'

    print(f"  {th:>10.2f} {macro_l1_val:>10.4f} {m2:>10.4f} {n_l2:>10}{marker}")

print(f"\n  最佳 margin threshold = {best_th}，驗證集 Macro = {best_macro:.4f}")

# ============================================================
# STEP 7：驗證集混淆對確認
# ============================================================

print(f"\n[STEP 7] 驗證集混淆對確認")
cls_list   = sorted(np.unique(y_val_true))
n_cls      = len(cls_list)
cls_to_idx = {c: i for i, c in enumerate(cls_list)}

cm_val = np.zeros((n_cls, n_cls), dtype=int)
for t, p in zip(y_val_true, y_val_l1):
    cm_val[cls_to_idx[t]][cls_to_idx[p]] += 1

confusion_pairs = []
for i in range(n_cls):
    for j in range(n_cls):
        if i != j and cm_val[i][j] > 0:
            ratio = cm_val[i][j] / cm_val[i].sum()
            confusion_pairs.append((cls_list[i], cls_list[j],
                                    cm_val[i][j], ratio))

confusion_pairs.sort(key=lambda x: -x[3])
print(f"  前 5 大混淆對：")
for true_c, pred_c, n_err, ratio in confusion_pairs[:5]:
    pair = tuple(sorted([true_c, pred_c]))
    flag = '✓ 預設' if pair in PREDEFINED_PAIRS else ''
    print(f"    {true_c} → {pred_c}：{n_err:>4} 筆 ({ratio:.1%}) {flag}")

# ============================================================
# STEP 8：測試集最終評估（只跑一次）
# ============================================================

print(f"\n[STEP 8] 測試集最終評估（只跑一次）")

y_te_l1, te_margins, te_top2s = qda_predict_full(df_test, lda_pred_te)
y_te_l2, n_l2_te = apply_layer2(df_test, y_te_l1, te_margins,
                                 te_top2s, layer2_models, best_th)

y_true   = df_test['pitch_type'].values
pca_l1   = per_class_accuracy(y_true, y_te_l1)
pca_l2   = per_class_accuracy(y_true, y_te_l2)
macro_l1 = macro_accuracy(y_true, y_te_l1)
macro_l2 = macro_accuracy(y_true, y_te_l2)

print(f"\n  第二層觸發：{n_l2_te} / {len(df_test)} ({n_l2_te/len(df_test)*100:.1f}%)")
print(f"\n  {'球種':<8} {'第一層':>8} {'第二層':>8} {'變化':>8}")
print(f"  {'-'*36}")
for cls in sorted(pca_l2):
    a1 = pca_l1.get(cls, 0)
    a2 = pca_l2.get(cls, 0)
    diff = a2 - a1
    marker = '↑' if diff > 0.005 else ('↓' if diff < -0.005 else '─')
    print(f"  {cls:<8} {a1:>7.1%} {a2:>7.1%} {marker} {diff:>+.1%}")
print(f"  {'-'*36}")
print(f"  {'Macro':.<12} {macro_l1:>7.1%} {macro_l2:>7.1%}   {macro_l2-macro_l1:>+.1%}")

# ============================================================
# STEP 9：視覺化
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f'QDA + Confidence Margin + Layer 2 LDA\n'
    f'Layer 1 QDA={macro_l1:.1%}  Layer 2={macro_l2:.1%}  '
    f'(threshold={best_th})',
    fontsize=12
)

ths   = [r[0] for r in results]
m2s   = [r[2] for r in results]
n_l2s = [r[3] for r in results]

ax1 = axes[0]
ax1.plot(ths, m2s, 'o-', color='tomato', linewidth=2, label='L2 Macro')
ax1.axhline(macro_l1_val, color='steelblue', linestyle='--',
            linewidth=1.5, label=f'L1 Macro={macro_l1_val:.3f}')
ax1.axvline(best_th, color='gray', linestyle=':', linewidth=1.5,
            label=f'best={best_th}')
ax1.set_xlabel('Margin Threshold')
ax1.set_ylabel('Macro Accuracy (Validation)')
ax1.set_title('Margin Threshold Scan')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1b = ax1.twinx()
ax1b.bar(ths, n_l2s, width=0.03, alpha=0.2, color='orange', label='觸發數')
ax1b.set_ylabel('觸發筆數（驗證集）')

sorted_cls = sorted(pca_l2, key=lambda c: pca_l2[c], reverse=True)
x = np.arange(len(sorted_cls))
width = 0.35
axes[1].bar(x - width/2, [pca_l1[c] for c in sorted_cls], width,
            label=f'QDA L1 ({macro_l1:.1%})', color='steelblue', alpha=0.7)
axes[1].bar(x + width/2, [pca_l2[c] for c in sorted_cls], width,
            label=f'L2 LDA ({macro_l2:.1%})', color='tomato', alpha=0.7)
axes[1].axhline(macro_l1, color='steelblue', linestyle='--', linewidth=1)
axes[1].axhline(macro_l2, color='tomato', linestyle='--', linewidth=1)
axes[1].set_xticks(x)
axes[1].set_xticklabels(sorted_cls, rotation=30)
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Per-class Accuracy (Test)')
axes[1].legend()

cm = np.zeros((n_cls, n_cls), dtype=int)
for t, p in zip(y_true, y_te_l2):
    if p is not None:
        cm[cls_to_idx[t]][cls_to_idx[str(p)]] += 1
im = axes[2].imshow(cm, cmap='Blues')
axes[2].set_xticks(range(n_cls))
axes[2].set_yticks(range(n_cls))
axes[2].set_xticklabels(cls_list, rotation=45, ha='right', fontsize=8)
axes[2].set_yticklabels(cls_list, fontsize=8)
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')
axes[2].set_title('Confusion Matrix (Test)')
if n_cls <= 12:
    for i in range(n_cls):
        for j in range(n_cls):
            val = cm[i][j]
            if val > 0:
                color = 'white' if val > cm.max() * 0.5 else 'black'
                axes[2].text(j, i, str(val), ha='center', va='center',
                             fontsize=7, color=color)
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig('qda_hierarchical_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n圖表已儲存：qda_hierarchical_results.png")

# ============================================================
# STEP 10：摘要
# ============================================================

print("\n" + "=" * 60)
print("  QDA 階層式分類器摘要")
print("=" * 60)
print(f"  資料切割          ：60/20/20")
print(f"  LDA 左右投（驗證）：{lda_acc_val*100:.2f}%")
print(f"  最佳 margin 門檻  ：{best_th}（驗證集選出）")
print(f"  測試集第二層觸發  ：{n_l2_te} / {len(df_test)} ({n_l2_te/len(df_test)*100:.1f}%)")
print(f"  第一層 QDA Macro  ：{macro_l1:.4f} ({macro_l1:.1%})")
print(f"  第二層 LDA Macro  ：{macro_l2:.4f} ({macro_l2:.1%})")
print(f"  提升幅度          ：{macro_l2-macro_l1:+.4f}")
print("=" * 60)
print("""
QDA vs NB 說明：
  NB 假設特徵條件獨立（covariance 對角矩陣）
  QDA 估計每個球種的完整 covariance matrix
  → 捕捉特徵相關性（球速×轉速等）
  → 決策邊界為二次曲線，更有彈性
  → 後驗機率 P(Ci|x) 有真正的機率意義

信心評估說明：
  margin = P(top1|x) - P(top2|x)
  margin 小 → 兩個球種後驗機率接近 → 真正不確定
  margin 大 → 第一名壓倒性領先 → 直接使用
  threshold 在驗證集上選定，不接觸測試集
""")


# SL margin 診斷
sl_mask = y_true == 'SL'
sl_margins = te_margins[sl_mask]
sl_l1 = y_te_l1[sl_mask]

print("\n[SL Margin 診斷]")
print(f"  SL 總筆數：{sl_mask.sum()}")
for target in ['FC', 'ST', 'CU']:
    wrong_mask = sl_l1 == target
    if wrong_mask.sum() == 0:
        continue
    margins_wrong = sl_margins[wrong_mask]
    print(f"\n  SL → {target}（{wrong_mask.sum()} 筆）")
    print(f"    margin 平均：{margins_wrong.mean():.3f}")
    print(f"    margin < 0.3：{(margins_wrong < 0.3).sum()} 筆")
    print(f"    margin < 0.5：{(margins_wrong < 0.5).sum()} 筆")
    print(f"    margin > 0.5：{(margins_wrong > 0.5).sum()} 筆")
