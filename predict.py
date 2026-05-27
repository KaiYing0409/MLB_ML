"""
predict.py
==========
載入訓練好的模型，對單筆投球物理特徵進行球種預測。

使用前請確認：
  1. 已執行 step2_feature_engineering.py（產出 global_ff_baseline.pkl）
  2. 已執行 train_and_save.py（產出 model.pkl）

使用方式：
  python predict.py
"""

import numpy as np
import pickle

# ============================================================
# 模型 Class 定義（pickle 還原需要）
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
# 載入模型與基準線
# ============================================================

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

lda_hand      = model['lda_hand']
lda_mu        = model['lda_mu']
lda_sig       = model['lda_sig']
lda_feats     = model['lda_feats']
qda_R         = model['qda_R']
mu_R          = model['mu_R']
sig_R         = model['sig_R']
qda_L         = model['qda_L']
mu_L          = model['mu_L']
sig_L         = model['sig_L']
layer2_models = model['layer2_models']
best_th       = model['best_th']
pitch_feats   = model['pitch_feats']

# ============================================================
# 工具函數
# ============================================================

def zscore_transform(X, mu, sig):
    return (X - mu) / sig

def compute_features(raw: dict) -> dict:
    """
    輸入原始物理特徵，計算 spin_axis sin/cos 轉換。
    """
    feats = dict(raw)
    rad = np.radians(raw['spin_axis'])
    feats['spin_axis_sin'] = np.sin(rad)
    feats['spin_axis_cos'] = np.cos(rad)
    return feats

def predict_pitch(raw: dict) -> dict:
    """
    主預測函數。
    輸入：原始物理特徵 dict
    輸出：預測結果 dict（球種、margin、是否觸發第二層）
    """
    feats = compute_features(raw)

    # --- LDA 左右投 ---
    x_lda = np.array([feats[f] for f in lda_feats], dtype=float).reshape(1, -1)
    x_lda_z = zscore_transform(x_lda, lda_mu, lda_sig)
    hand = lda_hand.predict(x_lda_z)[0]

    # --- QDA 球種分類 ---
    x_pitch = np.array([feats[f] for f in pitch_feats], dtype=float).reshape(1, -1)

    if hand == 'R':
        x_z = zscore_transform(x_pitch, mu_R, sig_R)
        probs = qda_R.posterior(x_z)[0]
        classes = qda_R.classes_
    else:
        x_z = zscore_transform(x_pitch, mu_L, sig_L)
        probs = qda_L.posterior(x_z)[0]
        classes = qda_L.classes_

    top2_idx = np.argsort(probs)[-2:][::-1]
    pred_l1  = classes[top2_idx[0]]
    top2     = classes[top2_idx[1]]
    margin   = probs[top2_idx[0]] - probs[top2_idx[1]]

    # --- 第二層 BinaryLDA ---
    layer2_triggered = False
    pred_final = pred_l1

    if margin < best_th:
        pair = tuple(sorted([pred_l1, top2]))
        if pair in layer2_models:
            lda2, mu2, sig2, feats2 = layer2_models[pair]
            x2 = np.array([feats[f] for f in feats2], dtype=float).reshape(1, -1)
            pred_final = lda2.predict(zscore_transform(x2, mu2, sig2))[0]
            layer2_triggered = True

    return {
        'predicted_pitch': pred_final,
        'margin':          round(float(margin), 4),
        'hand':            hand,
        'layer2':          layer2_triggered,
        'top2_candidate':  top2,
    }

# ============================================================
# 球種代碼對照表
# ============================================================

PITCH_NAMES = {
    'FF': '四縫線速球 Four-Seam Fastball',
    'SI': '伸卡球 Sinker',
    'FC': '卡特球 Cutter',
    'SL': '滑球 Slider',
    'ST': '橫掃滑球 Sweeper',
    'CU': '曲球 Curveball',
    'CH': '變速球 Changeup',
    'FS': '分叉球 Splitter',
}

# ============================================================
# 輸入介面
# ============================================================

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  請輸入數字。")

def main():
    print("\n" + "=" * 55)
    print("  棒球球種分類器 — 單筆預測")
    print("=" * 55)
    print("請輸入投球物理特徵（來源：MLB Statcast）\n")

    raw = {
        'release_speed':              get_float("  球速 release_speed (mph)："),
        'release_spin_rate':          get_float("  轉速 release_spin_rate (rpm)："),
        'spin_axis':                  get_float("  旋轉軸 spin_axis (0-360°)："),
        'api_break_x_arm':            get_float("  水平位移 api_break_x_arm (英吋)："),
        'api_break_z_with_gravity':   get_float("  垂直位移 api_break_z_with_gravity (英吋)："),
        'pfx_x':                      get_float("  水平位移 pfx_x (英吋)："),
        'ax':                         get_float("  水平加速度 ax (ft/s²)："),
        'vx0':                        get_float("  水平初速 vx0 (ft/s)："),
        'ay':                         get_float("  縱向加速度 ay (ft/s²)："),
        'vy0':                        get_float("  縱向初速 vy0 (ft/s)："),
        'arm_angle':                  get_float("  手臂角度 arm_angle (度)："),
    }

    result = predict_pitch(raw)

    pitch_code = result['predicted_pitch']
    pitch_name = PITCH_NAMES.get(pitch_code, pitch_code)

    print("\n" + "=" * 55)
    print("  預測結果")
    print("=" * 55)
    print(f"  球種        ：{pitch_code}  {pitch_name}")
    print(f"  信心分數    ：{result['margin']:.4f}  ", end="")
    if result['margin'] >= 0.5:
        print("（高信心）")
    elif result['margin'] >= 0.2:
        print("（中信心）")
    else:
        print("（低信心，結果僅供參考）")
    print(f"  投手慣用手  ：{result['hand']}")
    print(f"  第二層觸發  ：{'是（混淆對修正）' if result['layer2'] else '否'}")
    if result['layer2']:
        print(f"  修正前候選  ：{result['top2_candidate']}")
    print("=" * 55)

if __name__ == '__main__':
    main()
