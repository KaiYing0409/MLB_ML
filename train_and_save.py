"""
train_and_save.py
=================
訓練完整的階層式分類器並將所有模型參數存成 model.pkl。

執行完畢後會產生：
  model.pkl            — 所有分類器參數
  global_ff_baseline.pkl — 全聯盟 FF 基準線（若 Step 2 已存則跳過）

使用方式：
  python train_and_save.py

之後用 predict.py 進行單筆預測，不需要重新訓練。
"""

import numpy as np
import pandas as pd
import pickle

# ============================================================
# 設定
# ============================================================

DATA_PATH   = 'testdata_relative_phy.csv'
MIN_SAMPLES = 200
RANDOM_SEED = 42
VAL_RATIO   = 0.2
TEST_RATIO  = 0.2

PITCH_FEATURES = [
    'api_break_x_arm',
    'api_break_z_with_gravity',
    'spin_axis_sin',
    'spin_axis_cos',
    'rel_release_speed',
    'rel_release_spin_rate',
    'ay',
    'vy0',
]

LDA_FEATURES = [
    'pfx_x', 'ax', 'vx0', 'spin_axis_sin', 'spin_axis_cos',
    'arm_angle', 'rel_api_break_x_arm',
    'rel_release_speed', 'rel_release_spin_rate',
]

PREDEFINED_PAIRS = {
    ('CH', 'FS'): {'feats': ['rel_release_spin_rate', 'rel_api_break_x_arm', 'spin_axis_sin', 'rel_api_break_z_with_gravity'], 'weighted': False},
    ('FC', 'SL'): {'feats': ['rel_api_break_z_with_gravity', 'rel_release_speed', 'spin_axis_cos', 'rel_api_break_x_arm'], 'weighted': True},
    ('SL', 'ST'): {'feats': ['rel_api_break_x_arm', 'rel_release_speed', 'spin_axis_cos', 'rel_release_spin_rate'], 'weighted': True},
    ('CU', 'ST'): {'feats': ['rel_api_break_z_with_gravity', 'spin_axis_cos', 'rel_release_speed', 'rel_api_break_x_arm'], 'weighted': True},
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

# ============================================================
# QDA 分類器
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
            Si  += np.eye(X.shape[1]) * 1e-4
            Si_inv   = np.linalg.inv(Si)
            log_det  = np.log(np.linalg.det(Si) + 1e-300)
            log_prior = np.log(Nc / N)
            self.params_[c] = {
                'mi': mi, 'Si_inv': Si_inv,
                'log_det': log_det, 'log_prior': log_prior,
            }

    def posterior(self, X):
        n = len(X)
        K = len(self.classes_)
        G = np.zeros((n, K))
        for j, c in enumerate(self.classes_):
            p    = self.params_[c]
            diff = X - p['mi']
            maha = np.sum(diff @ p['Si_inv'] * diff, axis=1)
            G[:, j] = -0.5 * p['log_det'] - 0.5 * maha + p['log_prior']
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
# BinaryLDA
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
        N_pos, N_neg = len(X_pos), len(X_neg)
        if weighted:
            self.threshold = (N_pos * z_pos.mean() + N_neg * z_neg.mean()) / (N_pos + N_neg)
        else:
            self.threshold = (z_pos.mean() + z_neg.mean()) / 2

    def predict(self, X):
        return np.where(X @ self.w >= self.threshold,
                        self.class_pos, self.class_neg)

# ============================================================
# STEP 1：讀資料 + 切割
# ============================================================

print("=" * 55)
print("  Train and Save — 階層式分類器訓練")
print("=" * 55)

print("\n[1] 讀資料")
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

np.random.seed(RANDOM_SEED)
idx      = np.random.permutation(len(df))
n_test   = int(len(df) * TEST_RATIO)
n_val    = int(len(df) * VAL_RATIO)
df_train = df.iloc[idx[n_test + n_val:]].reset_index(drop=True)
df_val   = df.iloc[idx[n_test:n_test + n_val]].reset_index(drop=True)

print(f"  訓練集：{len(df_train):,}  驗證集：{len(df_val):,}")

# ============================================================
# STEP 2：訓練 LDA 左右投
# ============================================================

print("\n[2] 訓練 LDA 左右投分類器")
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
lda_acc_val  = np.mean(lda_pred_val == df_val['p_throws'].values)
print(f"  驗證集 LDA 準確率：{lda_acc_val*100:.2f}%")

# ============================================================
# STEP 3：訓練 QDA
# ============================================================

print("\n[3] 訓練 QDA 球種分類器")
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

print("\n[4] 訓練第二層 BinaryLDA")
layer2_models = {}
for pair, config in PREDEFINED_PAIRS.items():
    class_a, class_b = pair
    feats       = [f for f in config['feats'] if f in df.columns]
    if len(feats) < 2:
        continue
    mask    = df_train['pitch_type'].isin([class_a, class_b])
    df_pair = df_train[mask].reset_index(drop=True)
    X = df_pair[feats].values.astype(float)
    y = df_pair['pitch_type'].values
    mu, sig = zscore_fit(X)
    lda2 = BinaryLDA()
    lda2.fit(zscore_transform(X, mu, sig), y,
             class_pos=class_a, class_neg=class_b,
             weighted=config['weighted'])
    layer2_models[pair] = (lda2, mu, sig, feats)
    print(f"  {class_a} vs {class_b}：J(w)={((X[y==class_a] @ lda2.w).mean() - (X[y==class_b] @ lda2.w).mean())**2:.2f}")

# ============================================================
# STEP 5：驗證集掃描 margin threshold
# ============================================================

print("\n[5] 驗證集掃描 margin threshold")

def qda_predict_full(df_eval, lda_pred):
    n = len(df_eval)
    y_l1    = np.empty(n, dtype=object)
    margins = np.empty(n, dtype=float)
    top2s   = np.empty(n, dtype=object)
    idx_R   = np.where(lda_pred == 'R')[0]
    idx_L   = np.where(lda_pred == 'L')[0]
    if len(idx_R) > 0:
        X_R = df_eval.iloc[idx_R][pitch_feats].values.astype(float)
        p, m, t = qda_R.predict_with_margin(zscore_transform(X_R, mu_R, sig_R))
        y_l1[idx_R], margins[idx_R], top2s[idx_R] = p, m, t
    if len(idx_L) > 0:
        X_L = df_eval.iloc[idx_L][pitch_feats].values.astype(float)
        p, m, t = qda_L.predict_with_margin(zscore_transform(X_L, mu_L, sig_L))
        y_l1[idx_L], margins[idx_L], top2s[idx_L] = p, m, t
    return y_l1, margins, top2s

def apply_layer2(df_eval, y_l1, margins, top2s, margin_th):
    y_final = y_l1.copy()
    for i in range(len(df_eval)):
        if margins[i] >= margin_th:
            continue
        pair = tuple(sorted([y_l1[i], top2s[i]]))
        if pair not in layer2_models:
            continue
        lda2, mu2, sig2, feats2 = layer2_models[pair]
        x = df_eval.iloc[i][feats2].values.astype(float).reshape(1, -1)
        y_final[i] = lda2.predict(zscore_transform(x, mu2, sig2))[0]
    return y_final

y_val_l1, val_margins, val_top2s = qda_predict_full(df_val, lda_pred_val)
y_val_true = df_val['pitch_type'].values

best_macro = 0
best_th    = None
print(f"  {'threshold':>10} {'Macro':>10}")
for th in MARGIN_LIST:
    y_l2  = apply_layer2(df_val, y_val_l1, val_margins, val_top2s, th)
    macro = macro_accuracy(y_val_true, y_l2)
    marker = ''
    if macro > best_macro:
        best_macro = macro
        best_th    = th
        marker = ' ← 最佳'
    print(f"  {th:>10.2f} {macro:>10.4f}{marker}")

print(f"\n  最佳 threshold = {best_th}，驗證集 Macro = {best_macro:.4f}")

# ============================================================
# STEP 6：存出 model.pkl
# ============================================================

print("\n[6] 儲存模型參數")

model = {
    # LDA 左右投
    'lda_hand':      lda_hand,
    'lda_mu':        lda_mu,
    'lda_sig':       lda_sig,
    'lda_feats':     lda_feats,
    # QDA 右投
    'qda_R':         qda_R,
    'mu_R':          mu_R,
    'sig_R':         sig_R,
    # QDA 左投
    'qda_L':         qda_L,
    'mu_L':          mu_L,
    'sig_L':         sig_L,
    # 第二層
    'layer2_models': layer2_models,
    'best_th':       best_th,
    # 特徵名稱（predict.py 需要）
    'pitch_feats':   pitch_feats,
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("  已儲存：model.pkl")
print(f"\n{'='*55}")
print(f"  訓練完成")
print(f"  LDA 左右投（驗證）：{lda_acc_val*100:.2f}%")
print(f"  最佳 margin 門檻  ：{best_th}")
print(f"  驗證集 Macro      ：{best_macro:.4f}")
print(f"{'='*55}")
print("\n接下來執行 predict.py 進行單筆預測。")
