# -*- coding: cp950 -*-
"""
predict.py
==========
嚙????嚙踝蕭??蝺游末???璅∴蕭??嚙??嚙????嚙踝蕭??????????嚙踝蕭????嚙賢噩??嚙踝蕭?????蝔殷蕭??皜穿蕭??

雿選蕭?嚙踝蕭??嚙??蝣綽蕭??嚙??
  1. 撌莎蕭?嚙踝蕭?? step2_feature_engineering.py嚙????嚙踝蕭?? global_ff_baseline.pkl嚙??
  2. 撌莎蕭?嚙踝蕭?? train_and_save.py嚙????嚙踝蕭?? model.pkl嚙??

雿選蕭?嚙踝蕭?嚙踝蕭??嚙??
  python predict.py
"""

'''
input

=======================================================
  嚙????????蝔殷蕭??嚙????? ??? ??嚙踝蕭?????嚙??
=======================================================
嚙??頛賂蕭?嚙踝蕭???????嚙踝蕭????嚙賢噩嚙??嚙??嚙??嚙??MLB Statcast嚙??

  ?????? release_speed (mph)嚙??84
  嚙????? release_spin_rate (rpm)嚙??2000
  ???嚙??嚙?? spin_axis (0-360簞)嚙??210
  瘞游像嚙??嚙?? api_break_x_arm (??嚙踝蕭??)嚙??-8.2
  ?????嚙踝蕭??嚙?? api_break_z_with_gravity (??嚙踝蕭??)嚙??28.5
  瘞游像嚙??嚙?? pfx_x (??嚙踝蕭??)嚙??-6.1
  瘞游像??????嚙?? ax (ft/s簡)嚙??-8.4
  瘞游像?????? vx0 (ft/s)嚙??-6.2
  蝮梧蕭????????嚙?? ay (ft/s簡)嚙??27.3
  蝮梧蕭???????? vy0 (ft/s)嚙??-138.1
  ??????嚙??嚙?? arm_angle (嚙??)嚙??28.5
  ???????????嚙踝蕭?? p_throws (R/L)嚙??R

output
=======================================================
  ???皜穿蕭?????
=======================================================
  ???嚙??        嚙??SI  隡賂蕭?嚙踝蕭?? Sinker
  靽∴蕭????????    嚙??1.0000  嚙??嚙??靽∴蕭??嚙??
  ???????????嚙踝蕭??  嚙??L
  蝚穿蕭??撅方孛???  嚙?????
=======================================================
(base) zhengkaiying@macbookairdrop 嚙????嚙賢飛嚙??final_project % 
'''
import numpy as np
import pickle

# ============================================================
# 璅∴蕭?? Class 嚙??蝢抬蕭??pickle ?????????嚙??嚙??
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
# 頛???交芋????????箸??鞈????
# ============================================================
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model.pkl'
FF_BASELINE_PATH = BASE_DIR / 'global_ff_baseline.pkl'

# Some pickled objects were saved when BinaryLDA/QDAClassifier were defined
# in __main__. Make them available there so pickle can resolve the classes.
sys.modules['__main__'].QDAClassifier = QDAClassifier
sys.modules['__main__'].BinaryLDA = BinaryLDA

# Lazy-load model: 嘗試載入 pickle，但若因為 numpy C-extension 缺失或環境不正確導致失敗，
# 會輸出友善提示並建立一組最小的 fallback 物件以避免整個程式崩潰。
MODEL_AVAILABLE = False
model = None
ff_baseline = None

def ensure_model_loaded():
    global MODEL_AVAILABLE, model, ff_baseline
    global lda_hand, lda_mu, lda_sig, lda_feats
    global qda_R, mu_R, sig_R, qda_L, mu_L, sig_L
    global layer2_models, best_th, pitch_feats

    if MODEL_AVAILABLE:
        return

    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        with open(FF_BASELINE_PATH, 'rb') as f:
            ff_baseline = pickle.load(f)

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

        MODEL_AVAILABLE = True
        print("模型載入完成。")

    except ModuleNotFoundError as e:
        print("\n[Error] 載入 pickled model 時發生 ModuleNotFoundError：", e)
        print("這通常表示目前 Python 環境的 NumPy 安裝不完整或版本不相容 (缺少 numpy._core)。")
        print("請在終端執行下面其中一個指令修復 NumPy：\n  - pip: python -m pip install --upgrade --force-reinstall numpy\n  - conda: conda install -y numpy")
        print("或執行專案根目錄中的 reinstall_numpy.bat / reinstall_numpy.ps1。程式將使用暫時的 fallback 模型繼續執行。\n")

        # Minimal fallback implementations so程式不會 crash
        class DummyQDA:
            def __init__(self):
                self.classes_ = np.array(['FF'])
            def posterior(self, X):
                n = len(X)
                return np.ones((n, 1))
            def predict(self, X):
                return np.array(['FF'] * len(X))

        class DummyLDA:
            def predict(self, X):
                # default to right-hand
                return np.array(['R'] * len(X))

        lda_hand = DummyLDA()
        lda_mu = np.zeros(1)
        lda_sig = np.ones(1)
        lda_feats = []

        qda_R = DummyQDA()
        mu_R = np.zeros(1)
        sig_R = np.ones(1)

        qda_L = DummyQDA()
        mu_L = np.zeros(1)
        sig_L = np.ones(1)

        layer2_models = {}
        best_th = 0.0
        pitch_feats = []

        # FF baseline minimal
        ff_baseline = {
            'release_speed': 0.0,
            'release_spin_rate': 0.0,
            'api_break_x_arm': 0.0,
            'api_break_z_with_gravity': 0.0,
        }

        MODEL_AVAILABLE = False

    except Exception as e:
        print("\n[Error] 載入模型時發生例外：", e)
        print("請確認 'model.pkl' 與 'global_ff_baseline.pkl' 存在且與當前環境相容。")
        # 設置同樣的 fallback
        lda_hand = type('L', (), {'predict': lambda self, X: np.array(['R'] * len(X))})()
        lda_mu = np.zeros(1)
        lda_sig = np.ones(1)
        lda_feats = []
        qda_R = type('Q', (), {'posterior': lambda self, X: np.ones((len(X),1)), 'classes_': np.array(['FF'])})()
        mu_R = np.zeros(1); sig_R = np.ones(1)
        qda_L = qda_R; mu_L = mu_R; sig_L = sig_R
        layer2_models = {}
        best_th = 0.0
        pitch_feats = []
        ff_baseline = {'release_speed':0.0,'release_spin_rate':0.0,'api_break_x_arm':0.0,'api_break_z_with_gravity':0.0}
        MODEL_AVAILABLE = False

    # end ensure_model_loaded


# ============================================================
# 撌伐蕭?嚙踝蕭?嚙踝蕭??
# ============================================================

def zscore_transform(X, mu, sig):
    return (X - mu) / sig

def compute_features(raw: dict) -> dict:
    """
    頛賂蕭?嚙踝蕭??嚙????嚙踝蕭????嚙賢噩嚙??頛賂蕭?嚙踝蕭??嚙????嚙踝蕭????????嚙????嚙踝蕭?嚙賢噩 dict???
    ?????嚙踝蕭????嚙踝蕭????嚙賢噩嚙??嚙?????spin_axis sin/cos 嚙????????
    """
    feats = dict(raw)

    # ??嚙踝蕭????嚙賢噩嚙??隞伐蕭?嚙踝蕭?嚙踝蕭?? FF ?????嚙踝蕭?嚙踝蕭?嚙踝蕭??嚙??
    for col in ['release_speed', 'release_spin_rate',
                'api_break_x_arm', 'api_break_z_with_gravity']:
        feats[f'rel_{col}'] = raw[col] - ff_baseline[col]

    # spin_axis ??嚙踝蕭????嚙踝蕭?????
    rad = np.radians(raw['spin_axis'])
    feats['spin_axis_sin'] = np.sin(rad)
    feats['spin_axis_cos'] = np.cos(rad)

    return feats

def predict_pitch(raw: dict) -> dict:
    """
    銝鳴蕭??皜穿蕭?嚙踝蕭?嚙踝蕭??
    頛賂蕭?嚙踝蕭?????嚙????嚙踝蕭????嚙賢噩 dict
    頛賂蕭?嚙踝蕭?????皜穿蕭????? dict嚙?????蝔殷蕭??margin?????嚙踝蕭?嚙質孛??嚙賜洵嚙??撅歹蕭??
    """
    # 確保模型已載入（或已經設置 fallback）
    ensure_model_loaded()

    feats = compute_features(raw)

    # --- LDA 撌佗蕭?嚙踝蕭?? ---
    x_lda = np.array([feats[f] for f in lda_feats], dtype=float).reshape(1, -1)
    x_lda_z = zscore_transform(x_lda, lda_mu, lda_sig)
    hand = lda_hand.predict(x_lda_z)[0]

    # --- QDA ???蝔殷蕭??嚙?? ---
    x_pitch = np.array([feats[f] for f in pitch_feats], dtype=float).reshape(1, -1)

    if hand == 'R':
        x_z = zscore_transform(x_pitch, mu_R, sig_R)
        probs = qda_R.posterior(x_z)[0]
        classes = qda_R.classes_
    else:
        x_z = zscore_transform(x_pitch, mu_L, sig_L)
        probs = qda_L.posterior(x_z)[0]
        classes = qda_L.classes_

    # 安全處理：若模型僅包含單一 class，避免取 top2 時發生 out-of-bounds
    if len(classes) == 1:
        pred_l1 = classes[0]
        top2 = None
        margin = 1.0
    else:
        top2_idx = np.argsort(probs)[-2:][::-1]
        pred_l1  = classes[top2_idx[0]]
        top2     = classes[top2_idx[1]]
        margin   = probs[top2_idx[0]] - probs[top2_idx[1]]

    # --- 蝚穿蕭??嚙?? BinaryLDA ---
    layer2_triggered = False
    pred_final = pred_l1

    if (top2 is not None) and (margin < best_th):
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
# ???蝔桐誨蝣潘蕭????嚙質”
# ============================================================

PITCH_NAMES = {
    'FF': '???蝮恬蕭???????? Four-Seam Fastball',
    'SI': '隡賂蕭?嚙踝蕭?? Sinker',
    'FC': '??嚙踝蕭?嚙踝蕭?? Cutter',
    'SL': '嚙????? Slider',
    'ST': '璈恬蕭??嚙????? Sweeper',
    'CU': '??嚙踝蕭?? Curveball',
    'CH': '嚙???????? Changeup',
    'FS': '????????? Splitter',
}

# ============================================================
# 頛賂蕭?嚙踝蕭?????
# ============================================================

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  嚙??頛賂蕭?嚙踝蕭?嚙踝蕭?????")

def get_hand(prompt):
    while True:
        val = input(prompt).strip().upper()
        if val in ['R', 'L']:
            return val
        print("  嚙??頛賂蕭?? R嚙????嚙踝蕭??嚙????? L嚙??撌佗蕭??嚙?????")

def main():
    print("\n" + "=" * 55)
    print("  嚙????????蝔殷蕭??嚙????? ??? ??嚙踝蕭?????嚙??")
    print("=" * 55)
    print("嚙??頛賂蕭?嚙踝蕭???????嚙踝蕭????嚙賢噩嚙??嚙??嚙??嚙??MLB Statcast嚙??\n")

    raw = {
        'release_speed':              get_float("  ?????? release_speed (mph)嚙??"),
        'release_spin_rate':          get_float("  嚙????? release_spin_rate (rpm)嚙??"),
        'spin_axis':                  get_float("  ???嚙??嚙?? spin_axis (0-360簞)嚙??"),
        'api_break_x_arm':            get_float("  瘞游像嚙??嚙?? api_break_x_arm (??嚙踝蕭??)嚙??"),
        'api_break_z_with_gravity':   get_float("  ?????嚙踝蕭??嚙?? api_break_z_with_gravity (??嚙踝蕭??)嚙??"),
        'pfx_x':                      get_float("  瘞游像嚙??嚙?? pfx_x (??嚙踝蕭??)嚙??"),
        'ax':                         get_float("  瘞游像??????嚙?? ax (ft/s簡)嚙??"),
        'vx0':                        get_float("  瘞游像?????? vx0 (ft/s)嚙??"),
        'ay':                         get_float("  蝮梧蕭????????嚙?? ay (ft/s簡)嚙??"),
        'vy0':                        get_float("  蝮梧蕭???????? vy0 (ft/s)嚙??"),
        'arm_angle':                  get_float("  ??????嚙??嚙?? arm_angle (嚙??)嚙??"),
        'p_throws':                   get_hand( "  ???????????嚙踝蕭?? p_throws (R/L)嚙??"),
    }

    result = predict_pitch(raw)

    pitch_code = result['predicted_pitch']
    pitch_name = PITCH_NAMES.get(pitch_code, pitch_code)

    print("\n" + "=" * 55)
    print("  ???皜穿蕭?????")
    print("=" * 55)
    print(f"  ???嚙??        嚙??{pitch_code}  {pitch_name}")
    print(f"  靽∴蕭????????    嚙??{result['margin']:.4f}  ", end="")
    if result['margin'] >= 0.5:
        print("嚙??嚙??靽∴蕭??嚙??")
    elif result['margin'] >= 0.2:
        print("嚙??銝凋縑嚙??嚙??")
    else:
        print("嚙??嚙??靽∴蕭??嚙??嚙????????嚙????????嚙??")
    print(f"  ???????????嚙踝蕭??  嚙??{result['hand']}")
    print(f"  蝚穿蕭??撅方孛???  嚙??{'??嚙踝蕭??瘛瘀蕭??嚙??靽格迤嚙??' if result['layer2'] else '???'}")
    if result['layer2']:
        print(f"  靽格迤?????????  嚙??{result['top2_candidate']}")
    print("=" * 55)

if __name__ == '__main__':
    main()
