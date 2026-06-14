import json
import pickle
import numpy as np
import pandas as pd

TRAIN_PATH  = 'data_train.csv'
VAL_PATH    = 'data_val.csv'
OUTPUT_JSON = 'features.json'
OUTPUT_PKL  = 'model.pkl'

CORR_THRESH  = 0.9
ELBOW_THRESH = 0.003   # 驗證集準確率提升 < 0.3% 視為不顯著

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

ALL_FEATS = [
    'release_speed', 'effective_speed',
    'release_spin_rate',
    'spin_axis_sin', 'spin_axis_cos',
    'pfx_x', 'pfx_z',
    'api_break_z_with_gravity',
    'vx0', 'vy0', 'vz0',
    'ax', 'ay', 'az',
]

# QDA（forward selection 驗證用）
class QDAClassifier:
    def fit(self, X, y):
        self.classes_  = np.unique(y)
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

# 工具函式
def zscore(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1e-9)

def compute_f_ratio(x_scaled, y, groups):
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
    X    = df_data[feat_list].values.astype(float)
    mu   = X.mean(axis=0)
    sig  = X.std(axis=0) + 1e-9
    corr = np.corrcoef(((X - mu) / sig).T)

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

def val_acc_macro(df_tr, df_va, feats, label_col, classes):
    df_tr = df_tr.dropna(subset=feats)
    df_va = df_va.dropna(subset=feats)
    X_tr  = df_tr[feats].values.astype(float)
    y_tr  = df_tr[label_col].values
    X_va  = df_va[feats].values.astype(float)
    y_va  = df_va[label_col].values

    mu  = X_tr.mean(axis=0)
    sig = X_tr.std(axis=0) + 1e-9
    qda = QDAClassifier()
    qda.fit((X_tr - mu) / sig, y_tr)
    y_pred = qda.predict((X_va - mu) / sig)

    per_cls = [
        (y_pred[y_va == c] == c).mean()
        for c in classes if (y_va == c).sum() > 0
    ]
    return np.mean(per_cls)

def forward_select(df_tr, df_va, feat_order, label_col, classes, elbow_thresh, hand_label=''):
    print(f"\n  ── {hand_label}  Train={len(df_tr):,}，Val={len(df_va):,} ──")
    print(f"  {'N':<5} {'加入特徵':<30} {'Val Acc':>10}")
    print(f"  {'-'*48}")

    acc_history = []
    elbow_n     = 1

    for n_feat in range(1, len(feat_order) + 1):
        current_feats = feat_order[:n_feat]
        added         = feat_order[n_feat - 1]

        min_samples = min((df_tr[label_col] == c).sum() for c in classes)
        if min_samples < 5:
            print(f"  {n_feat:<5} {added:<30} {'樣本不足':>10}")
            break

        acc = val_acc_macro(df_tr, df_va, current_feats, label_col, classes)
        acc_history.append((n_feat, added, acc))
        print(f"  {n_feat:<5} {added:<30} {acc:>10.1%}")

        if n_feat > 1:
            prev_acc = acc_history[-2][2]
            if acc - prev_acc >= elbow_thresh:
                elbow_n = n_feat

    return elbow_n, acc_history

# 載入資料
print("載入資料...")
df_train = pd.read_csv(TRAIN_PATH)
df_val   = pd.read_csv(VAL_PATH)

df_train['group'] = df_train['pitch_type'].map(GROUP_MAP)
df_val['group']   = df_val['pitch_type'].map(GROUP_MAP)

df_train = df_train.dropna(subset=['group']).reset_index(drop=True)
df_val   = df_val.dropna(subset=['group']).reset_index(drop=True)

avail = [f for f in ALL_FEATS if f in df_train.columns]
df_train = df_train.dropna(subset=avail).reset_index(drop=True)
df_val   = df_val.dropna(subset=avail).reset_index(drop=True)

df_tr_R = df_train[df_train['p_throws'] == 'R'].reset_index(drop=True)
df_tr_L = df_train[df_train['p_throws'] == 'L'].reset_index(drop=True)
df_va_R = df_val[df_val['p_throws'] == 'R'].reset_index(drop=True)
df_va_L = df_val[df_val['p_throws'] == 'L'].reset_index(drop=True)

print(f"Train — 右投：{len(df_tr_R):,} 筆，左投：{len(df_tr_L):,} 筆")
print(f"Val   — 右投：{len(df_va_R):,} 筆，左投：{len(df_va_L):,} 筆")

# STEP 1：相關係數篩選（F-ratio 以 train 資料計算）
print(f"\n{'='*65}")
print(f"  STEP 1：相關係數篩選（|r| > {CORR_THRESH}，左右投分開，取聯集）")
print(f"{'='*65}")

fr_R_global = {f: compute_f_ratio(zscore(df_tr_R[f].values), df_tr_R['group'].values, GROUPS) for f in avail}
fr_L_global = {f: compute_f_ratio(zscore(df_tr_L[f].values), df_tr_L['group'].values, GROUPS) for f in avail}

kept_R, log_R = corr_filter(df_tr_R, avail, fr_R_global, CORR_THRESH)
kept_L, log_L = corr_filter(df_tr_L, avail, fr_L_global, CORR_THRESH)

removed_all = set(d for _, _, _, _, d in log_R) | set(d for _, _, _, _, d in log_L)

for hand_label, log in [('右投 R', log_R), ('左投 L', log_L)]:
    print(f"\n  ── {hand_label} ──")
    if log:
        print(f"  {'特徵A':<26} {'特徵B':<26} {'r':>7}  {'保留':<24} 移除")
        print(f"  {'-'*95}")
        for fA, fB, r, keep, discard in log:
            print(f"  {fA:<26} {fB:<26} {r:>7.3f}  {keep:<24} {discard}")
    else:
        print("  （無高相關特徵對）")

filtered_feats = [f for f in avail if f not in removed_all]
print(f"\n  移除（{len(removed_all)} 個）：{sorted(removed_all)}")
print(f"  保留（{len(filtered_feats)} 個）：{filtered_feats}")

# STEP 2：全體 F-ratio 排序（三大類，左右投分開，train only）
print(f"\n{'='*65}")
print(f"  STEP 2：全體 F-ratio（三大類，train 資料）")
print(f"{'='*65}")

l1_fr_R = {f: compute_f_ratio(zscore(df_tr_R[f].values), df_tr_R['group'].values, GROUPS) for f in filtered_feats}
l1_fr_L = {f: compute_f_ratio(zscore(df_tr_L[f].values), df_tr_L['group'].values, GROUPS) for f in filtered_feats}

l1_feat_order = sorted(filtered_feats, key=lambda f: l1_fr_R[f], reverse=True)

print(f"\n  {'排名':<5} {'特徵':<26} {'F-ratio R':>11} {'F-ratio L':>11}")
print('  ' + '-' * 56)
for i, f in enumerate(l1_feat_order):
    print(f"  {i+1:<5} {f:<26} {l1_fr_R[f]:>11.1f} {l1_fr_L[f]:>11.1f}")

# STEP 3：各大類 F-ratio 排序（train only）
print(f"\n{'='*65}")
print(f"  STEP 3：各大類內部 F-ratio（train 資料）")
print(f"{'='*65}")

sub_feat_order = {}

for grp, pitches in PITCH_IN_GROUP.items():
    mask_R = df_train['pitch_type'].isin(pitches) & (df_train['p_throws'] == 'R')
    mask_L = df_train['pitch_type'].isin(pitches) & (df_train['p_throws'] == 'L')
    dg_R   = df_train[mask_R]
    dg_L   = df_train[mask_L]

    fr_R = {f: compute_f_ratio(zscore(dg_R[f].values), dg_R['pitch_type'].values, pitches) for f in filtered_feats}
    fr_L = {f: compute_f_ratio(zscore(dg_L[f].values), dg_L['pitch_type'].values, pitches) for f in filtered_feats}

    order = sorted(filtered_feats, key=lambda f: fr_R[f], reverse=True)
    sub_feat_order[grp] = order

    print(f"\n  大類：{grp}  （{pitches}）")
    print(f"  {'排名':<5} {'特徵':<26} {'F R':>10} {'F L':>10}")
    print('  ' + '-' * 55)
    for i, f in enumerate(order):
        print(f"  {i+1:<5} {f:<26} {fr_R[f]:>10.1f} {fr_L[f]:>10.1f}")

# STEP 4：Layer 1 Forward Selection
print(f"\n{'='*65}")
print(f"  STEP 4：Layer 1 Forward Selection（三大類）")
print(f"{'='*65}")

l1_elbow_n = 1

for hand in ['R', 'L']:
    df_tr_h = df_train[df_train['p_throws'] == hand]
    df_va_h = df_val[df_val['p_throws'] == hand]

    elbow_n, history = forward_select(
        df_tr_h, df_va_h,
        l1_feat_order, 'group', GROUPS,
        ELBOW_THRESH, hand_label=f'{hand}投 Layer 1'
    )

    if hand == 'R':
        l1_elbow_n = elbow_n

    final_n     = l1_elbow_n
    final_feats = l1_feat_order[:final_n]
    final_acc   = history[final_n - 1][2] if history else None

    print(f"\n  → Elbow：TOP_{final_n}，特徵：{final_feats}")
    if final_acc:
        print(f"  → Val Acc：{final_acc:.1%}")

L1_FEATS = l1_feat_order[:l1_elbow_n]

# STEP 5：各大類 Forward Selection
print(f"\n{'='*65}")
print(f"  STEP 5：各大類 Forward Selection")
print(f"{'='*65}")

sub_feats_result = {}

for grp, pitches in PITCH_IN_GROUP.items():
    feat_order = sub_feat_order[grp]
    grp_elbow_n = 1

    print(f"\n{'='*55}")
    print(f"  大類：{grp}  （{pitches}）")

    for hand in ['R', 'L']:
        mask_tr = (df_train['group'] == grp) & (df_train['p_throws'] == hand)
        mask_va = (df_val['group']   == grp) & (df_val['p_throws']   == hand)
        df_tr_h = df_train[mask_tr]
        df_va_h = df_val[mask_va]

        elbow_n, history = forward_select(
            df_tr_h, df_va_h,
            feat_order, 'pitch_type', pitches,
            ELBOW_THRESH, hand_label=f'{hand}投 {grp}'
        )

        if hand == 'R':
            grp_elbow_n = elbow_n

        final_n     = grp_elbow_n
        final_feats = feat_order[:final_n]
        final_acc   = history[final_n - 1][2] if history else None

        print(f"\n  → Elbow：TOP_{final_n}，特徵：{final_feats}")
        if final_acc:
            print(f"  → Val Acc：{final_acc:.1%}")

        key = f'{grp}_{hand}' if grp == 'Offspeed' else grp
        sub_feats_result[key] = final_feats

# 摘要 & 輸出 features.json
print(f"\n{'='*65}")
print("  選用特徵摘要")
print(f"{'='*65}")
print(f"\n  Layer 1（{len(L1_FEATS)} 個）：{L1_FEATS}")
for key, val in sub_feats_result.items():
    if isinstance(val, list):
        print(f"  {key:<14}（{len(val)} 個）：{val}")
    else:
        print(f"  {key:<14}：直接輸出 {val}")

features = {
    'L1':          L1_FEATS,
    'Fastball':    sub_feats_result.get('Fastball', []),
    'Breaking':    sub_feats_result.get('Breaking', []),
    'Offspeed_R':  sub_feats_result.get('Offspeed_R', []),
    'Offspeed_L':  sub_feats_result.get('Offspeed_L', []),
}

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(features, f, ensure_ascii=False, indent=2)

print(f"\n已儲存：{OUTPUT_JSON}")


# 訓練最終模型並存 model.pkl
def fit_qda_final(df_tr, feats, label_col):
    df_tr = df_tr.dropna(subset=feats)
    X   = df_tr[feats].values.astype(float)
    y   = df_tr[label_col].values
    mu  = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-9
    qda = QDAClassifier()
    qda.fit((X - mu) / sig, y)
    return qda, mu, sig

print(f"\n{'='*65}")
print("  訓練最終模型")
print(f"{'='*65}")

l1_models = {}
for hand in ['R', 'L']:
    df_h = df_train[df_train['p_throws'] == hand]
    qda, mu, sig = fit_qda_final(df_h, L1_FEATS, 'group')
    l1_models[hand] = (qda, mu, sig)
    print(f"  L1 {hand}：{len(df_h):,} 筆，{len(L1_FEATS)} 個特徵")

l2_models = {}
SUB_FEATS = {
    'Fastball':   features['Fastball'],
    'Breaking':   features['Breaking'],
    'Offspeed_R': features['Offspeed_R'],
    'Offspeed_L': features['Offspeed_L'],
}
for grp, pitches in PITCH_IN_GROUP.items():
    for hand in ['R', 'L']:
        key   = f'{grp}_{hand}' if grp == 'Offspeed' else grp
        feats = SUB_FEATS[key]
        mask  = (df_train['group'] == grp) & (df_train['p_throws'] == hand)
        df_h  = df_train[mask]
        qda, mu, sig = fit_qda_final(df_h, feats, 'pitch_type')
        l2_models[f'{grp}_{hand}'] = (qda, mu, sig, feats)
        print(f"  L2 {grp}-{hand}：{len(df_h):,} 筆，{len(feats)} 個特徵")

model = {
    'l1_models':     l1_models,
    'l1_feats':      L1_FEATS,
    'l2_models':     l2_models,
    'group_map':     GROUP_MAP,
    'sub_feats':     SUB_FEATS,
    'lda_hand':      None, 'lda_mu': None, 'lda_sig': None, 'lda_feats': [],
    'qda_R':         None, 'mu_R': None, 'sig_R': None,
    'qda_L':         None, 'mu_L': None, 'sig_L': None,
    'layer2_models': {}, 'best_th': 0.0, 'pitch_feats': L1_FEATS,
}

with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(model, f)

print(f"\n已儲存：{OUTPUT_PKL}")
print("完成。")
