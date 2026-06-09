"""
Step 4 — Sequential Forward Selection（各大類子分類器）
========================================================
輸入：testdata_only_phy.csv
輸出：terminal — 各大類 elbow 點、選出特徵、驗證集準確率

流程：
  - 全資料切一次（60/20/20，seed=42）
  - 對每個大類（Fastball / Breaking / Offspeed）：
      左右投各自跑 Forward Selection
      依 F-ratio 排序逐步加特徵
      評估驗證集平均類別準確率
      找 elbow（右投準確率提升 < 0.3% 就停）
  - 左投 Offspeed 樣本極少，跳過（直接輸出 CH）

特徵（step3 篩選後，依各大類右投 F-ratio 排序）：
  Fastball：pfx_x, api_break_z_with_gravity, spin_axis_sin,
            release_speed, ay, release_spin_rate, spin_axis_cos, vx0, vz0
  Breaking：api_break_z_with_gravity, pfx_x, release_speed,
            spin_axis_cos, vz0, spin_axis_sin, release_spin_rate, vx0, ay
  Offspeed：release_spin_rate, pfx_x, api_break_z_with_gravity,
            vx0, spin_axis_sin, spin_axis_cos, release_speed, vz0, ay
"""

import numpy as np
import pandas as pd

INPUT_PATH = 'testdata_only_phy.csv'

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

# F-ratio 排序（step3 結果）
FEAT_ORDER = {
    'Fastball': [
        'pfx_x', 'api_break_z_with_gravity', 'spin_axis_sin',
        'release_speed', 'ay', 'release_spin_rate',
        'spin_axis_cos', 'vx0', 'vz0',
    ],
    'Breaking': [
        'api_break_z_with_gravity', 'pfx_x', 'release_speed',
        'spin_axis_cos', 'vz0', 'spin_axis_sin',
        'release_spin_rate', 'vx0', 'ay',
    ],
    'Offspeed': [
        'release_spin_rate', 'pfx_x', 'api_break_z_with_gravity',
        'vx0', 'spin_axis_sin', 'spin_axis_cos',
        'release_speed', 'vz0', 'ay',
    ],
}

ELBOW_THRESH = 0.003  # 提升 < 1% 視為不再顯著

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

def val_acc(df_tr, df_va, feats, pitches):
    """訓練 QDA 並回傳驗證集平均類別準確率"""
    df_tr = df_tr.dropna(subset=feats)
    df_va = df_va.dropna(subset=feats)
    X_tr = df_tr[feats].values.astype(float)
    y_tr = df_tr['pitch_type'].values
    X_va = df_va[feats].values.astype(float)
    y_va = df_va['pitch_type'].values

    mu  = X_tr.mean(axis=0)
    sig = X_tr.std(axis=0) + 1e-9
    X_tr_sc = (X_tr - mu) / sig
    X_va_sc = (X_va - mu) / sig

    qda = QDAClassifier()
    qda.fit(X_tr_sc, y_tr)
    y_pred = qda.predict(X_va_sc)

    per_cls = [
        (y_pred[y_va == pt] == pt).mean()
        for pt in pitches if (y_va == pt).sum() > 0
    ]
    return np.mean(per_cls)

# ============================================================
# 載入 & 切分
# ============================================================

print("載入資料...")
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
# Forward Selection
# ============================================================

results = {}  # {grp: {hand: {n, feats, acc}}}

for grp, pitches in PITCH_IN_GROUP.items():
    feat_order = FEAT_ORDER[grp]
    results[grp] = {}

    print(f"\n{'='*65}")
    print(f"  大類：{grp}  （球種：{pitches}）")
    print(f"  特徵加入順序：{feat_order}")
    print(f"{'='*65}")

    for hand in ['R', 'L']:

        # 左投 Offspeed 直接跳過
        if grp == 'Offspeed' and hand == 'L':
            print(f"\n  ── 左投 L：樣本極少，直接輸出 CH，跳過 ──")
            continue

        mask_tr  = (df_train['group'] == grp) & (df_train['p_throws'] == hand)
        mask_val = (df_val['group']   == grp) & (df_val['p_throws']   == hand)
        df_tr    = df_train[mask_tr]
        df_va    = df_val[mask_val]

        print(f"\n  ── {hand}投  Train={len(df_tr):,}，Val={len(df_va):,} ──")
        print(f"  {'特徵數':<6} {'加入特徵':<30} {'Val Acc':>10}")
        print(f"  {'-'*50}")

        acc_history = []
        elbow_n     = 1

        for n_feat in range(1, len(feat_order) + 1):
            current_feats = feat_order[:n_feat]
            added         = feat_order[n_feat - 1]

            # 確認每個球種在訓練集都有足夠樣本
            min_samples = min(
                (df_tr['pitch_type'] == pt).sum() for pt in pitches
            )
            if min_samples < 5:
                print(f"  {n_feat:<6} {added:<30} {'樣本不足':>10}")
                break

            acc = val_acc(df_tr, df_va, current_feats, pitches)
            acc_history.append((n_feat, added, acc))
            print(f"  {n_feat:<6} {added:<30} {acc:>10.1%}")

            # 更新 elbow：右投才用來決定停止點
            if hand == 'R' and n_feat > 1:
                prev_acc = acc_history[-2][2]
                if acc - prev_acc >= ELBOW_THRESH:
                    elbow_n = n_feat

        # 決定最終 TOP_N（右投用 elbow，左投用右投同樣的 N）
        if hand == 'R':
            final_n = elbow_n
            results[grp]['elbow_n'] = final_n

        final_n     = results[grp].get('elbow_n', len(feat_order))
        final_feats = feat_order[:final_n]
        final_acc   = acc_history[final_n - 1][2] if acc_history else None

        results[grp][hand] = {
            'n':     final_n,
            'feats': final_feats,
            'acc':   final_acc,
        }

        print(f"\n  → Elbow 點：TOP_{final_n}")
        print(f"  → 選出特徵：{final_feats}")
        if final_acc:
            print(f"  → Val Acc：{final_acc:.1%}")

# ============================================================
# 摘要
# ============================================================

print(f"\n{'='*65}")
print("  Forward Selection 結果摘要")
print(f"{'='*65}")
print(f"  {'大類':<12} {'手性':<6} {'TOP_N':>6} {'Val Acc':>10}  特徵")
print(f"  {'-'*75}")

for grp in PITCH_IN_GROUP:
    for hand in ['R', 'L']:
        if grp == 'Offspeed' and hand == 'L':
            print(f"  {grp:<12} {'L':<6} {'—':>6} {'—':>10}  直接輸出 CH")
            continue
        if hand not in results.get(grp, {}):
            continue
        res = results[grp][hand]
        print(f"  {grp:<12} {hand:<6} {res['n']:>6} {res['acc']:>10.1%}  {res['feats']}")

# ============================================================
# Layer 1 Forward Selection（三大類，左右投分開）
# ============================================================

L1_FEAT_ORDER = [
    'spin_axis_sin', 'api_break_z_with_gravity', 'release_speed',
    'pfx_x', 'spin_axis_cos', 'release_spin_rate', 'ay', 'vx0', 'vz0',
]
L1_GROUPS = ['Fastball', 'Breaking', 'Offspeed']

def val_acc_l1(df_tr, df_va, feats, groups):
    """Layer 1：訓練大類 QDA，回傳驗證集平均類別準確率"""
    df_tr = df_tr.dropna(subset=feats)
    df_va = df_va.dropna(subset=feats)
    X_tr = df_tr[feats].values.astype(float)
    y_tr = df_tr['group'].values
    X_va = df_va[feats].values.astype(float)
    y_va = df_va['group'].values

    mu  = X_tr.mean(axis=0)
    sig = X_tr.std(axis=0) + 1e-9
    X_tr_sc = (X_tr - mu) / sig
    X_va_sc = (X_va - mu) / sig

    qda = QDAClassifier()
    qda.fit(X_tr_sc, y_tr)
    y_pred = qda.predict(X_va_sc)

    per_cls = [
        (y_pred[y_va == g] == g).mean()
        for g in groups if (y_va == g).sum() > 0
    ]
    return np.mean(per_cls)

print(f"\n{'='*65}")
print("  Layer 1 Forward Selection（三大類，左右投分開）")
print(f"  特徵加入順序：{L1_FEAT_ORDER}")
print(f"{'='*65}")

l1_results = {}
l1_elbow_n = None

for hand in ['R', 'L']:
    df_tr_h = df_train[df_train['p_throws'] == hand]
    df_va_h = df_val[df_val['p_throws'] == hand]

    print(f"\n  ── {hand}投  Train={len(df_tr_h):,}，Val={len(df_va_h):,} ──")
    print(f"  {'特徵數':<6} {'加入特徵':<30} {'Val Acc':>10}")
    print(f"  {'-'*50}")

    acc_history = []
    elbow_n     = 1

    for n_feat in range(1, len(L1_FEAT_ORDER) + 1):
        current_feats = L1_FEAT_ORDER[:n_feat]
        added         = L1_FEAT_ORDER[n_feat - 1]

        acc = val_acc_l1(df_tr_h, df_va_h, current_feats, L1_GROUPS)
        acc_history.append((n_feat, added, acc))
        print(f"  {n_feat:<6} {added:<30} {acc:>10.1%}")

        if hand == 'R' and n_feat > 1:
            prev_acc = acc_history[-2][2]
            if acc - prev_acc >= ELBOW_THRESH:
                elbow_n = n_feat

    if hand == 'R':
        l1_elbow_n = elbow_n

    final_n     = l1_elbow_n if l1_elbow_n else len(L1_FEAT_ORDER)
    final_feats = L1_FEAT_ORDER[:final_n]
    final_acc   = acc_history[final_n - 1][2]

    l1_results[hand] = {'n': final_n, 'feats': final_feats, 'acc': final_acc}

    print(f"\n  → Elbow 點：TOP_{final_n}")
    print(f"  → 選出特徵：{final_feats}")
    print(f"  → Val Acc：{final_acc:.1%}")

print(f"\n{'='*65}")
print("  Layer 1 Forward Selection 結果摘要")
print(f"{'='*65}")
print(f"  {'手性':<6} {'TOP_N':>6} {'Val Acc':>10}  特徵")
print(f"  {'-'*75}")
for hand in ['R', 'L']:
    res = l1_results[hand]
    print(f"  {hand:<6} {res['n']:>6} {res['acc']:>10.1%}  {res['feats']}")