"""
predict.py
==========
載入訓練好的模型，對單筆投球物理特徵進行球種預測。

使用前請確認：
1. 已執行 step2_preprocess.py（產出 testdata_only_phy.csv）
2. 已執行 train_and_save.py（產出 model.pkl）

使用方式：
    python predict.py

或從其他程式 import：
    from predict import predict_pitch
    result = predict_pitch(raw_dict)
"""

import numpy as np
import pickle

# ============================================================
# QDAClassifier（pickle 還原需要與訓練時相同的 class 定義）
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
# 載入模型
# ============================================================

# ============================================================
# 模型載入（支援延遲載入與可用性查詢）
# ============================================================

MODEL_AVAILABLE = False
l1_models = None
l1_feats  = None
l2_models = None
group_map = None

def ensure_model_loaded(path='model.pkl'):
    """
    載入模型（若尚未載入）。
    final_v1.py 呼叫 predict.ensure_model_loaded() 時使用。
    """
    global MODEL_AVAILABLE, l1_models, l1_feats, l2_models, group_map
    if MODEL_AVAILABLE:
        return
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        l1_models = model['l1_models']
        l1_feats  = model['l1_feats']
        l2_models = model['l2_models']
        group_map = model['group_map']
        MODEL_AVAILABLE = True
        print(f"[predict] 模型載入成功：{path}")
    except FileNotFoundError:
        print(f"[predict] 警告：找不到 {path}，MODEL_AVAILABLE = False")
    except Exception as e:
        print(f"[predict] 警告：模型載入失敗（{e}），MODEL_AVAILABLE = False")

# 預設嘗試載入（直接執行或 import 時都會自動載入）
ensure_model_loaded()

GROUP_TO_PITCHES = {
    'Fastball': ['FF', 'SI', 'FC'],
    'Breaking': ['SL', 'ST', 'CU'],
    'Offspeed': ['CH', 'FS'],
}

# ============================================================
# 工具函數
# ============================================================

def compute_features(raw: dict) -> dict:
    """計算 spin_axis sin/cos，其餘欄位直接傳遞。"""
    feats = dict(raw)
    rad = np.radians(raw['spin_axis'])
    feats['spin_axis_sin'] = np.sin(rad)
    feats['spin_axis_cos'] = np.cos(rad)
    return feats

def predict_pitch(raw: dict) -> dict:
    """
    主預測函數。

    輸入 raw dict 必須包含：
        p_throws             : 'R' 或 'L'（投手慣用手）
        release_speed        : 球速 (mph)
        release_spin_rate    : 轉速 (rpm)
        spin_axis            : 旋轉軸 (0-360°)
        pfx_x                : 水平位移 (英吋)
        api_break_z_with_gravity : 垂直位移（含重力）
        vx0                  : 水平初速 (ft/s)
        vz0                  : 垂直初速 (ft/s)
        ay                   : 縱向加速度 (ft/s²)

    回傳 dict：
        predicted_pitch  : 預測球種代碼（如 'FF'）
        margin           : Layer 1 後驗機率差（top1 - top2）
        hand             : 投手慣用手（'R' 或 'L'）
        layer2           : False（新架構無第二層 LDA，保留此 key 維持合約）
        top2_candidate   : ''（同上）
    """
    if not MODEL_AVAILABLE:
        raise RuntimeError("模型未載入，請先執行 train_and_save.py 產生 model.pkl")
    feats = compute_features(raw)
    hand  = raw['p_throws']

    # --- Layer 1：三大類 QDA ---
    qda_l1, mu_l1, sig_l1 = l1_models[hand]
    x_l1 = np.array([feats[f] for f in l1_feats], dtype=float).reshape(1, -1)
    x_l1_z = (x_l1 - mu_l1) / sig_l1

    probs_l1 = qda_l1.posterior(x_l1_z)[0]
    top2_idx = np.argsort(probs_l1)[-2:][::-1]
    grp_pred = qda_l1.classes_[top2_idx[0]]
    margin   = float(probs_l1[top2_idx[0]] - probs_l1[top2_idx[1]])

    # --- Layer 2：子分類器 ---
    # 左投 Offspeed：直接輸出 CH
    if grp_pred == 'Offspeed' and hand == 'L':
        return {
            'predicted_pitch': 'CH',
            'margin':          round(margin, 4),
            'hand':            hand,
            'layer2':          False,
            'top2_candidate':  '',
        }

    key = f'{grp_pred}_R' if grp_pred == 'Offspeed' else grp_pred
    model_key = f'{grp_pred}_{hand}'

    qda_l2, mu_l2, sig_l2, l2_feats = l2_models[model_key]
    x_l2 = np.array([feats[f] for f in l2_feats], dtype=float).reshape(1, -1)
    x_l2_z = (x_l2 - mu_l2) / sig_l2

    pitch_pred = qda_l2.predict(x_l2_z)[0]

    return {
        'predicted_pitch': pitch_pred,
        'margin':          round(margin, 4),
        'hand':            hand,
        'layer2':          False,
        'top2_candidate':  '',
    }

# ============================================================
# 球種名稱對照
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
# Terminal 互動介面
# ============================================================

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  請輸入數字。")

def get_hand(prompt):
    while True:
        val = input(prompt).strip().upper()
        if val in ('R', 'L'):
            return val
        print("  請輸入 R 或 L。")

def main():
    print("\n" + "=" * 55)
    print("  棒球球種分類器 — 單筆預測")
    print("=" * 55)
    print("請輸入投球物理特徵（來源：MLB Statcast）\n")

    raw = {
        'p_throws':                  get_hand("  投手慣用手 p_throws (R/L)："),
        'release_speed':             get_float("  球速 release_speed (mph)："),
        'release_spin_rate':         get_float("  轉速 release_spin_rate (rpm)："),
        'spin_axis':                 get_float("  旋轉軸 spin_axis (0-360°)："),
        'pfx_x':                     get_float("  水平位移 pfx_x (英吋)："),
        'api_break_z_with_gravity':  get_float("  垂直位移 api_break_z_with_gravity (英吋)："),
        'vx0':                       get_float("  水平初速 vx0 (ft/s)："),
        'vz0':                       get_float("  垂直初速 vz0 (ft/s)："),
        'ay':                        get_float("  縱向加速度 ay (ft/s²)："),
    }

    result = predict_pitch(raw)
    pitch_code = result['predicted_pitch']
    pitch_name = PITCH_NAMES.get(pitch_code, pitch_code)

    print("\n" + "=" * 55)
    print("  預測結果")
    print("=" * 55)
    print(f"  球種     ：{pitch_code} {pitch_name}")
    print(f"  信心分數  ：{result['margin']:.4f}", end=" ")
    if result['margin'] >= 0.5:
        print("（高信心）")
    elif result['margin'] >= 0.2:
        print("（中信心）")
    else:
        print("（低信心，結果僅供參考）")
    print(f"  投手慣用手：{result['hand']}")
    print("=" * 55)

if __name__ == '__main__':
    main()
