import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TEST_PATH  = 'data_test.csv'
MODEL_PATH = 'model.pkl'

PITCH_ORDER = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS']
PITCH_NAMES = {
    'FF': 'Four-Seam Fastball',
    'SI': 'Sinker',
    'FC': 'Cutter',
    'SL': 'Slider',
    'ST': 'Sweeper',
    'CU': 'Curveball',
    'CH': 'Changeup',
    'FS': 'Splitter',
}

GROUP_MAP = {
    'FF': 'Fastball', 'SI': 'Fastball', 'FC': 'Fastball',
    'SL': 'Breaking', 'ST': 'Breaking', 'CU': 'Breaking',
    'CH': 'Offspeed', 'FS': 'Offspeed',
}

# QDA
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

    def posterior(self, X):
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

# 載入資料 & 模型
print("載入測試資料...")
df_test = pd.read_csv(TEST_PATH)
df_test['group'] = df_test['pitch_type'].map(GROUP_MAP)
df_test = df_test.dropna(subset=['group']).reset_index(drop=True)
print(f"測試集：{len(df_test):,} 筆")
print(df_test['pitch_type'].value_counts().reindex(PITCH_ORDER).to_string())

print("\n載入模型...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

l1_models = model['l1_models']
l1_feats  = model['l1_feats']
l2_models = model['l2_models']


# Pipeline 預測
def pipeline_predict(df_input):
    preds = np.empty(len(df_input), dtype=object)

    for hand in ['R', 'L']:
        mask_h = (df_input['p_throws'] == hand).values
        if mask_h.sum() == 0:
            continue

        df_h  = df_input[mask_h]
        idx_h = np.where(mask_h)[0]

        qda_l1, mu_l1, sig_l1 = l1_models[hand]
        X_l1     = df_h[l1_feats].values.astype(float)
        grp_pred = qda_l1.predict((X_l1 - mu_l1) / sig_l1)

        for grp in ['Fastball', 'Breaking', 'Offspeed']:
            mask_g = grp_pred == grp
            if mask_g.sum() == 0:
                continue

            df_g  = df_h[mask_g]
            idx_g = idx_h[mask_g]

            entry = l2_models[f'{grp}_{hand}']
            qda_l2, mu_l2, sig_l2, feats = entry
            X_l2         = df_g[feats].values.astype(float)
            preds[idx_g] = qda_l2.predict((X_l2 - mu_l2) / sig_l2)

    return preds

print("\n預測中...")
y_pred = pipeline_predict(df_test)
y_true = df_test['pitch_type'].values

# 混淆矩陣計算
cls_idx = {c: i for i, c in enumerate(PITCH_ORDER)}
cm = np.zeros((8, 8), dtype=int)
for t, p in zip(y_true, y_pred):
    if t in cls_idx and p in cls_idx:
        cm[cls_idx[t], cls_idx[p]] += 1

# 輸出
per_cls_acc = {}
for pt in PITCH_ORDER:
    mask = y_true == pt
    if mask.sum() == 0:
        continue
    per_cls_acc[pt] = (y_pred[mask] == pt).mean()

macro = np.mean(list(per_cls_acc.values()))

print(f"\n{'='*68}")
print(f"  測試集結果")
print(f"{'='*68}")
print(f"\n  {'球種':<8} {'名稱':<24} {'準確率':>10} {'N':>8}  {'主要混淆'}")
print(f"  {'-'*68}")

for pt in PITCH_ORDER:
    if pt not in per_cls_acc:
        continue
    i   = cls_idx[pt]
    n   = cm[i].sum()
    acc = per_cls_acc[pt]

    row = cm[i].copy()
    row[i] = 0
    top_err_idx = np.argmax(row)
    top_err_n   = row[top_err_idx]
    top_err_pt  = PITCH_ORDER[top_err_idx]
    confusion_str = f"→ {top_err_pt} ({top_err_n:,})" if top_err_n > 0 else ''

    print(f"  {pt:<8} {PITCH_NAMES[pt]:<24} {acc:>10.1%} {n:>8,}  {confusion_str}")

print(f"\n  Macro 準確率：{macro:.1%}")

print(f"\n  {'球種':<8} {'Precision':>10} {'Recall':>10} {'F1':>8}")
print(f"  {'-'*40}")
for i, pt in enumerate(PITCH_ORDER):
    if cm[i].sum() == 0:
        continue
    tp   = cm[i, i]
    fp   = cm[:, i].sum() - tp
    fn   = cm[i, :].sum() - tp
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    print(f"  {pt:<8} {prec:>10.1%} {rec:>10.1%} {f1:>8.1%}")

# 圖：全體混淆矩陣（顯示百分比，以每列總數為分母）
row_sums = cm.sum(axis=1, keepdims=True)
cm_pct   = np.where(row_sums > 0, cm / row_sums, 0.0)

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, shrink=0.8, format='%.0%%')
ax.set_xticks(range(8))
ax.set_yticks(range(8))
ax.set_xticklabels(PITCH_ORDER, fontsize=10)
ax.set_yticklabels(PITCH_ORDER, fontsize=10)
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_title(f'Test Set — Overall Confusion Matrix\nMacro Accuracy: {macro:.1%}',
             fontsize=12, fontweight='bold')
for i in range(8):
    for j in range(8):
        pct = cm_pct[i, j]
        if pct == 0:
            continue
        color = 'white' if pct > 0.5 else 'black'
        ax.text(j, i, f'{pct:.1%}', ha='center', va='center', fontsize=8, color=color)
plt.tight_layout()
plt.savefig('confusion_matrix_overall.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n已儲存：confusion_matrix_overall.png")
print(f"完成。測試集 Macro 準確率：{macro:.1%}")
