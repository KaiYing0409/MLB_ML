"""
Step 5 — 完整分類 Pipeline 端對端評估
======================================
輸入：testdata_only_phy.csv
輸出：terminal 評估結果（驗證集 + 測試集）

架構：
  Layer 1：三大類 QDA（左右投各一）
    特徵（9）：spin_axis_sin, api_break_z_with_gravity, release_speed,
               pfx_x, spin_axis_cos, release_spin_rate, ay, vx0, vz0

  Layer 2：子分類器（依大類 routing）
    Fastball R/L（3）：pfx_x, api_break_z_with_gravity, spin_axis_sin
    Breaking R/L（9）：api_break_z_with_gravity, pfx_x, release_speed,
                       spin_axis_cos, vz0, spin_axis_sin, release_spin_rate,
                       vx0, ay
    Offspeed R  （8）：release_spin_rate, pfx_x, api_break_z_with_gravity,
                       vx0, spin_axis_sin, spin_axis_cos, release_speed, vz0
    Offspeed L        ：直接輸出 CH

資料切分：60/20/20，seed=42（全資料切一次）
評估指標：各球種準確率、Macro 準確率（各球種平均）、混淆矩陣
"""

import numpy as np
import pandas as pd

INPUT_PATH  = 'testdata_only_phy.csv'
PITCH_ORDER = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS']

GROUP_MAP = {
    'FF': 'Fastball', 'SI': 'Fastball', 'FC': 'Fastball',
    'SL': 'Breaking', 'ST': 'Breaking', 'CU': 'Breaking',
    'CH': 'Offspeed', 'FS': 'Offspeed',
}

PITCH_IN_GROUP = {
    'Fastball': ['FF', 'SI', 'FC'],
    'Breaking': ['SL', 'ST', 'CU'],
    'Offspeed': ['CH', 'FS'],
}

L1_FEATS = [
    'spin_axis_sin', 'api_break_z_with_gravity', 'release_speed',
    'pfx_x', 'spin_axis_cos', 'release_spin_rate', 'ay', 'vx0', 'vz0',
]

SUB_FEATS = {
    'Fastball': [
        'pfx_x', 'api_break_z_with_gravity', 'spin_axis_sin',
    ],
    'Breaking': [
        'api_break_z_with_gravity', 'pfx_x', 'release_speed',
        'spin_axis_cos', 'vz0', 'spin_axis_sin',
        'release_spin_rate', 'vx0', 'ay',
    ],
    'Offspeed_R': [
        'release_spin_rate', 'pfx_x', 'api_break_z_with_gravity',
        'vx0', 'spin_axis_sin', 'spin_axis_cos', 'release_speed', 'vz0',
    ],
}

# ============================================================
# QDA
# ============================================================

class QDAClassifier:
    def __init__(self):
        self.classes_  = None
        self.priors_   = {}
        self.means_    = {}
        self.inv_covs_ = {}
        self.log_dets_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        N = len(y)
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = len(Xc) / N
            self.means_[c]  = Xc.mean(axis=0)
            cov = np.cov(Xc.T, ddof=1)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            cov += np.eye(cov.shape[0]) * 1e-6
            self.inv_covs_[c] = np.linalg.inv(cov)
            _, logdet = np.linalg.slogdet(cov)
            self.log_dets_[c] = logdet

    def predict(self, X):
        scores = np.zeros((len(X), len(self.classes_)))
        for ci, c in enumerate(self.classes_):
            diff = X - self.means_[c]
            quad = np.sum(diff @ self.inv_covs_[c] * diff, axis=1)
            scores[:, ci] = (
                -0.5 * quad
                - 0.5 * self.log_dets_[c]
                + np.log(self.priors_[c])
            )
        return self.classes_[np.argmax(scores, axis=1)]

# ============================================================
# 訓練工具
# ============================================================

def fit_qda(df_tr, feats, label_col):
    """訓練 QDA，回傳 (model, mu, sig)"""
    df_tr = df_tr.dropna(subset=feats)
    X  = df_tr[feats].values.astype(float)
    y  = df_tr[label_col].values
    mu  = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-9
    qda = QDAClassifier()
    qda.fit((X - mu) / sig, y)
    return qda, mu, sig

# ============================================================
# 讀資料 & 切分
# ============================================================

print("讀資料...")
df = pd.read_csv(INPUT_PATH)
df['group'] = df['pitch_type'].map(GROUP_MAP)
df = df.dropna(subset=['group']).reset_index(drop=True)
print(f"總筆數：{len(df):,}")

rng = np.random.RandomState(42)
idx = np.arange(len(df))
rng.shuffle(idx)
n    = len(idx)
n_tr = int(n * 0.6)
n_va = int(n * 0.2)

df_train = df.iloc[idx[:n_tr]].reset_index(drop=True)
df_val   = df.iloc[idx[n_tr:n_tr + n_va]].reset_index(drop=True)
df_test  = df.iloc[idx[n_tr + n_va:]].reset_index(drop=True)
print(f"Train：{len(df_train):,}，Val：{len(df_val):,}，Test：{len(df_test):,}")

# ============================================================
# 訓練所有模型（僅用訓練集）
# ============================================================

print("\n訓練模型...")
models = {}

# Layer 1
for hand in ['R', 'L']:
    df_h = df_train[df_train['p_throws'] == hand]
    qda, mu, sig = fit_qda(df_h, L1_FEATS, 'group')
    models[f'L1_{hand}'] = (qda, mu, sig)
    print(f"  L1 {hand}：{len(df_h):,} 筆")

# Layer 2
for grp, pitches in PITCH_IN_GROUP.items():
    for hand in ['R', 'L']:
        if grp == 'Offspeed' and hand == 'L':
            continue
        feat_key = f'{grp}_R' if grp == 'Offspeed' else grp
        feats    = SUB_FEATS[feat_key]
        mask     = (df_train['group'] == grp) & (df_train['p_throws'] == hand)
        df_h     = df_train[mask]
        qda, mu, sig = fit_qda(df_h, feats, 'pitch_type')
        models[f'L2_{grp}_{hand}'] = (qda, mu, sig, feats)
        print(f"  L2 {grp}-{hand}：{len(df_h):,} 筆")

# ============================================================
# Pipeline 預測
# ============================================================

def pipeline_predict(df_input):
    preds = np.empty(len(df_input), dtype=object)

    for hand in ['R', 'L']:
        mask_h = (df_input['p_throws'] == hand).values
        if mask_h.sum() == 0:
            continue

        df_h  = df_input[mask_h]
        idx_h = np.where(mask_h)[0]

        # Layer 1
        qda_l1, mu_l1, sig_l1 = models[f'L1_{hand}']
        X_l1     = df_h[L1_FEATS].values.astype(float)
        grp_pred = qda_l1.predict((X_l1 - mu_l1) / sig_l1)

        # Layer 2 routing
        for grp in ['Fastball', 'Breaking', 'Offspeed']:
            mask_g = grp_pred == grp
            if mask_g.sum() == 0:
                continue

            df_g  = df_h[mask_g]
            idx_g = idx_h[mask_g]

            if grp == 'Offspeed' and hand == 'L':
                preds[idx_g] = 'CH'
                continue

            feat_key           = f'{grp}_R' if grp == 'Offspeed' else grp
            qda_l2, mu_l2, sig_l2, feats = models[f'L2_{grp}_{hand}']
            X_l2               = df_g[feats].values.astype(float)
            preds[idx_g]       = qda_l2.predict((X_l2 - mu_l2) / sig_l2)

    return preds

# ============================================================
# 評估函數
# ============================================================

def evaluate(y_true, y_pred, label):
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")

    per_cls_acc = {}
    for pt in PITCH_ORDER:
        mask = y_true == pt
        if mask.sum() == 0:
            continue
        per_cls_acc[pt] = (y_pred[mask] == pt).mean()

    macro = np.mean(list(per_cls_acc.values()))

    print(f"\n  {'球種':<8} {'準確率':>10} {'N':>8}")
    print(f"  {'-'*30}")
    for pt in PITCH_ORDER:
        if pt in per_cls_acc:
            n = (y_true == pt).sum()
            print(f"  {pt:<8} {per_cls_acc[pt]:>10.1%} {n:>8,}")
    print(f"\n  Macro 準確率：{macro:.1%}")

    # 混淆矩陣
    cls_idx = {c: i for i, c in enumerate(PITCH_ORDER)}
    cm = np.zeros((8, 8), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in cls_idx and p in cls_idx:
            cm[cls_idx[t], cls_idx[p]] += 1

    print(f"\n  混淆矩陣（列=實際，行=預測）：")
    print(f"  {'':>8}" + ''.join(f"{pt:>8}" for pt in PITCH_ORDER))
    for i, pt in enumerate(PITCH_ORDER):
        print(f"  {pt:>8}" + ''.join(f"{cm[i,j]:>8}" for j in range(8)))

    # Precision / Recall
    print(f"\n  {'球種':<8} {'Precision':>10} {'Recall':>10} {'N':>8}")
    print(f"  {'-'*40}")
    for i, pt in enumerate(PITCH_ORDER):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        n  = cm[i, :].sum()
        if n == 0:
            continue
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        print(f"  {pt:<8} {prec:>10.1%} {rec:>10.1%} {n:>8,}")

    return macro

# ============================================================
# 驗證集 & 測試集評估
# ============================================================

print("\n評估...")

y_val_pred  = pipeline_predict(df_val)
y_val_true  = df_val['pitch_type'].values
val_macro   = evaluate(y_val_true, y_val_pred, '驗證集')

y_test_pred = pipeline_predict(df_test)
y_test_true = df_test['pitch_type'].values
test_macro  = evaluate(y_test_true, y_test_pred, '測試集')

print(f"\n{'='*62}")
print(f"  最終結果")
print(f"{'='*62}")
print(f"  驗證集 Macro：{val_macro:.1%}")
print(f"  測試集 Macro：{test_macro:.1%}")
