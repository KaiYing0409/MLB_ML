"""
train_and_save.py
=================
訓練階層式 QDA 分類器並存成 model.pkl。

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

輸入：testdata_only_phy.csv
輸出：model.pkl
"""

import numpy as np
import pandas as pd
import pickle

INPUT_PATH  = 'testdata_only_phy.csv'
OUTPUT_PATH = 'model.pkl'

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
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        N = len(y)
        self.priors_   = {}
        self.means_    = {}
        self.inv_covs_ = {}
        self.log_dets_ = {}
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

    def posterior(self, X):
        """回傳後驗機率（供 predict_pitch 的 margin 計算用）"""
        scores = np.zeros((len(X), len(self.classes_)))
        for ci, c in enumerate(self.classes_):
            diff = X - self.means_[c]
            quad = np.sum(diff @ self.inv_covs_[c] * diff, axis=1)
            scores[:, ci] = (
                -0.5 * quad
                - 0.5 * self.log_dets_[c]
                + np.log(self.priors_[c])
            )
        scores -= scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

# ============================================================
# 訓練工具
# ============================================================

def fit_qda(df_tr, feats, label_col):
    df_tr = df_tr.dropna(subset=feats)
    X  = df_tr[feats].values.astype(float)
    y  = df_tr[label_col].values
    mu  = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-9
    qda = QDAClassifier()
    qda.fit((X - mu) / sig, y)
    return qda, mu, sig

# ============================================================
# 載入資料
# ============================================================

print("載入資料...")
df = pd.read_csv(INPUT_PATH)
df['group'] = df['pitch_type'].map(GROUP_MAP)
df = df.dropna(subset=['group']).reset_index(drop=True)
print(f"總筆數：{len(df):,}")

# 全資料用訓練集（60%）訓練，與 step5_pipeline.py 保持一致
rng = np.random.RandomState(42)
idx = np.arange(len(df))
rng.shuffle(idx)
n_tr = int(len(idx) * 0.6)
df_train = df.iloc[idx[:n_tr]].reset_index(drop=True)
print(f"訓練集：{len(df_train):,} 筆")

# ============================================================
# 訓練 Layer 1
# ============================================================

print("\n訓練 Layer 1（三大類 QDA）...")
l1_models = {}
for hand in ['R', 'L']:
    df_h = df_train[df_train['p_throws'] == hand]
    qda, mu, sig = fit_qda(df_h, L1_FEATS, 'group')
    l1_models[hand] = (qda, mu, sig)
    print(f"  L1 {hand}：{len(df_h):,} 筆")

# ============================================================
# 訓練 Layer 2
# ============================================================

print("\n訓練 Layer 2（子分類器）...")
l2_models = {}
for grp, pitches in PITCH_IN_GROUP.items():
    for hand in ['R', 'L']:
        if grp == 'Offspeed' and hand == 'L':
            print(f"  L2 Offspeed-L：直接輸出 CH，跳過訓練")
            continue
        feat_key = f'{grp}_R' if grp == 'Offspeed' else grp
        feats    = SUB_FEATS[feat_key]
        mask     = (df_train['group'] == grp) & (df_train['p_throws'] == hand)
        df_h     = df_train[mask]
        qda, mu, sig = fit_qda(df_h, feats, 'pitch_type')
        l2_models[f'{grp}_{hand}'] = (qda, mu, sig, feats)
        print(f"  L2 {grp}-{hand}：{len(df_h):,} 筆，{len(feats)} 個特徵")

# ============================================================
# 存檔
# ============================================================

model = {
    # 新架構
    'l1_models':  l1_models,   # {'R': (qda, mu, sig), 'L': (qda, mu, sig)}
    'l1_feats':   L1_FEATS,
    'l2_models':  l2_models,   # {'Fastball_R': (qda, mu, sig, feats), ...}
    'group_map':  GROUP_MAP,
    'sub_feats':  SUB_FEATS,

    # 保留舊 key（避免其他程式讀 pkl 時 KeyError）
    'lda_hand':     None,
    'lda_mu':       None,
    'lda_sig':      None,
    'lda_feats':    [],
    'qda_R':        None,
    'mu_R':         None,
    'sig_R':        None,
    'qda_L':        None,
    'mu_L':         None,
    'sig_L':        None,
    'layer2_models': {},
    'best_th':      0.0,
    'pitch_feats':  L1_FEATS,
}

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"\n已儲存：{OUTPUT_PATH}")
print("完成。")
