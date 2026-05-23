# MLB_ML

# 棒球球種分類器（Baseball Pitch Type Classifier）

利用 MLB Statcast 投球物理數據，透過階層式分類架構（QDA + BinaryLDA）對球種進行自動分類。

---

## 專案結構

```
├── README.md
├── step1_preprocess_eda.py       # 資料前處理 + EDA + F-ratio 特徵分析
├── step2_feature_engineering.py  # 相對特徵工程（投手基準線對齊 + spin_axis 轉換）
├── step3_classifier.py           # 主分類器（LDA → QDA → BinaryLDA 階層式架構）
└── data/
    ├── statcast_bat_tracking_2024_2025.csv   # 原始資料（需自行下載，見下方說明）
    ├── testdata_only_phy.csv                 # Step 1 產出（鏡像處理後，10萬筆）
    └── testdata_relative_phy.csv             # Step 2 產出（相對特徵，主分類器吃這個）
```

---

## 快速開始

### 1. 安裝環境

```bash
pip install numpy pandas matplotlib scipy
```

### 2. 取得資料

原始資料來源：[MLB Statcast + Bat Tracking (Kaggle)](https://www.kaggle.com/)

下載後將 `statcast_bat_tracking_2024_2025.csv` 放到專案根目錄。

### 3. 依序執行三支腳本

```bash
python step1_preprocess_eda.py      # 約 2-5 分鐘
python step2_feature_engineering.py # 約 1-2 分鐘
python step3_classifier.py          # 約 3-5 分鐘
```

---

## 資料處理流程（Step 1）

### 使用的物理特徵欄位

| 類別 | 欄位名稱 | 說明 |
|------|---------|------|
| 球速 | `release_speed` | 出手球速 (mph) |
| 球速 | `effective_speed` | 有效球速（含出手延伸修正） |
| 旋轉 | `release_spin_rate` | 轉速 (rpm) |
| 旋轉 | `spin_axis` | 旋轉軸方向 (0–360°) |
| 位移 | `api_break_x_arm` | 水平位移（投手手臂方向，英吋） |
| 位移 | `api_break_z_with_gravity` | 垂直位移（含重力，英吋） |
| 位移 | `pfx_x`, `pfx_z` | 水平/垂直位移（去除重力） |
| 加速度 | `ax`, `ay`, `az` | 加速度三分量 |
| 出手 | `release_pos_x/y/z` | 出手點座標 |
| 其他 | `p_throws` | 投手慣用手（L/R），用於左右投分類 |

### 資料清理步驟

**Step 1.1 — 球種篩選**

保留佔全資料比例 ≥ 1% 的球種，去除樣本數不足的罕見球種。

**Step 1.2 — 離群值移除**

對每個球種分別進行 IQR 清洗（k=3.0），避免跨球種的分布差異影響離群值判斷。

```
離群值定義：超出 [Q1 - 3×IQR, Q3 + 3×IQR] 範圍的數值
```

**Step 1.3 — KC → CU 合併**

指節曲球（Knuckle Curve, KC）和曲球（Curveball, CU）在物理特徵上高度相似，合併為同一類別以增加訓練樣本數。

**Step 1.4 — 鏡像處理（消除左右投雙峰）**

左投與右投在水平方向的物理量（pfx_x、ax、vx0、spin_axis）上呈現對稱鏡像分布，若不處理會造成同一球種在特徵空間中出現雙峰，嚴重影響分類器效果。

做法：將左投資料的水平方向特徵翻轉至右投視角：

```python
# 水平量直接取負號
df_L['pfx_x'] = -df_L['pfx_x']
df_L['ax']    = -df_L['ax']
df_L['vx0']   = -df_L['vx0']

# spin_axis 沿鏡像軸翻轉（鏡像軸 = 左右投均值的角平分線）
df_L['spin_axis'] = (2 * MIRROR_AXIS - df_L['spin_axis']) % 360
```

鏡像後，同一球種的分布從雙峰變為單峰，F-ratio 大幅提升。

**Step 1.5 — 分層抽樣**

從清理後的全量資料中，依球種比例抽取 10 萬筆，確保各球種比例維持一致。

產出：`testdata_only_phy.csv`（10萬筆，鏡像處理後）

---

## 特徵工程（Step 2）

### 相對特徵：以投手四縫線速球為基準

不同投手的球速、轉速等物理量有個體差異（例如 160 km/h 投手的變速球和 150 km/h 投手的變速球，絕對球速不同，但相對差距可能相似）。

做法：計算每位投手的四縫線速球（FF）平均值作為基準線，再對其他球種計算相對差值：

```
rel_release_speed = release_speed - 該投手的 FF 平均球速
rel_api_break_x_arm = api_break_x_arm - 該投手的 FF 平均水平位移
...以此類推
```

若該投手無 FF 資料，使用全聯盟 FF 均值替代。
樣本數低於 50 球的邊緣人投手直接剔除。

### spin_axis 週期性轉換

旋轉軸是 0–360° 的循環變數，直接使用數值會有邊界突變問題（例如 5° 和 355° 實際上只差 10°，但數值差 350°）。

做法：轉換為二維向量：

```python
spin_axis_sin = sin(spin_axis_rad)
spin_axis_cos = cos(spin_axis_rad)
```

### 最終進入分類器的特徵

| 特徵 | 說明 |
|------|------|
| `rel_api_break_x_arm` | 相對水平位移（vs 個人 FF） |
| `rel_api_break_z_with_gravity` | 相對垂直位移（vs 個人 FF） |
| `spin_axis_sin` | 旋轉軸正弦值 |
| `spin_axis_cos` | 旋轉軸餘弦值 |
| `rel_release_speed` | 相對球速（vs 個人 FF） |
| `rel_release_spin_rate` | 相對轉速（vs 個人 FF） |
| `ay` | 縱向加速度（原始值） |
| `vy0` | 縱向初速（原始值） |

產出：`testdata_relative_phy.csv`

---

## 分類器架構（Step 3）

### Pipeline 全貌

```
輸入：一筆投球的物理特徵向量
        │
        ▼
┌───────────────────────────────┐
│  第零層：LDA 左右投分類器      │  準確率 ~92%
│  BinaryLDA（Fisher's LDA）   │
│  特徵：pfx_x, ax, vx0,       │
│        spin_axis_sin/cos 等   │
└───────────────────────────────┘
        │ 右投 (R)          │ 左投 (L)
        ▼                   ▼
┌──────────────┐   ┌──────────────┐
│   QDA_R      │   │   QDA_L      │  第一層 Macro 82.2%
│  右投球種    │   │  左投球種    │
│  分類器      │   │  分類器      │
└──────────────┘   └──────────────┘
        │
        ▼
┌───────────────────────────────┐
│  信心評估                      │
│  margin = P(top1) - P(top2)   │
│  threshold = 0.5（驗證集選定） │
└───────────────────────────────┘
        │ margin ≥ 0.5           │ margin < 0.5（不確定）
        ▼                        ▼
  直接輸出第一層結果       ┌─────────────────────────┐
                           │  第二層：BinaryLDA       │
                           │  只針對已知混淆球種對    │
                           │  CH↔FS / FC↔SL / SL↔ST  │
                           │  CU↔ST                   │
                           └─────────────────────────┘
        │
        ▼
最終預測球種  ← 最終 Macro 83.3%
```

### 各層說明

**第零層：LDA 左右投分類**

Fisher's Linear Discriminant，二元分類左投/右投。由於左右投的水平物理量（pfx_x、vx0 等）有明顯的鏡像對稱性，LDA 能在這個方向上找到最大分離度的投影軸。

**第一層：QDA 球種分類**

Quadratic Discriminant Analysis，對左投/右投各建立獨立的分類器。

QDA 相比 Naive Bayes 的優勢在於：
- NB 假設特徵之間條件獨立（covariance 為對角矩陣）
- QDA 估計每個球種各自的完整 covariance matrix
- 可捕捉特徵之間的相關性（例如球速與轉速的負相關）
- 決策邊界為二次曲線，比線性邊界更有彈性

後驗機率公式（L6，Alpaydin 教材）：

```
g_i(x) = -1/2 * log|S_i| - 1/2 * (x-m_i)^T * S_i^{-1} * (x-m_i) + log P(C_i)
P(C_i|x) ∝ exp(g_i(x))
```

**信心評估**

```
margin = P(top1|x) - P(top2|x)
```

margin 越小，代表兩個候選球種的後驗機率越接近，模型越不確定。最佳 threshold（0.5）在驗證集上透過掃描選定，不接觸測試集。

**第二層：BinaryLDA 混淆對修正**

針對物理特徵上天然相似的球種對，各自訓練一個專屬的二元分類器。只有當第一層不確定（margin < threshold）且前兩名預測正好是其中一對時，才觸發第二層。

| 混淆對 | J(w) | 說明 |
|--------|------|------|
| CH ↔ FS | 1.67 | 變速球 vs 分叉球，球速接近 |
| FC ↔ SL | 2.21 | 卡特球 vs 滑球，水平位移相近 |
| SL ↔ ST | 1.91 | 滑球 vs 橫掃滑球，同屬滑球家族 |
| CU ↔ ST | 2.77 | 曲球 vs 橫掃滑球 |

### 資料切割

```
60% 訓練集 → 訓練所有模型參數
20% 驗證集 → 選 margin threshold、確認混淆對
20% 測試集 → 只跑一次，作為最終報告數字
```

### 最終測試集結果

| 球種 | 第一層 QDA | 第二層 LDA | 變化 |
|------|-----------|-----------|------|
| FF 四縫線速球 | 95.2% | 95.2% | — |
| SI 伸卡球 | 90.3% | 90.3% | — |
| CU 曲球 | 88.7% | 87.9% | ↓ -0.8% |
| CH 變速球 | 88.4% | 83.6% | ↓ -4.7% |
| ST 橫掃滑球 | 85.3% | 87.0% | ↑ +1.7% |
| FC 卡特球 | 83.7% | 86.0% | ↑ +2.3% |
| FS 分叉球 | 63.9% | 76.3% | ↑ +12.4% |
| SL 滑球 | 62.1% | 59.9% | ↓ -2.1% |
| **整體 Macro** | **82.2%** | **83.3%** | **+1.1%** |

準確率計算方式：各球種準確率的算術平均（Macro Accuracy）。

---

## 模型論述定位

本專案的架構屬於「基於混淆結構的探索式階層分類器（Confusion-aware Exploratory Hierarchical Classifier）」，對應以下教材概念：

| 模組 | 教材對應 |
|------|---------|
| QDA | L6 Multivariate Methods — Quadratic Discriminant |
| BinaryLDA | L10 Linear Discrimination — Fisher's LDA |
| margin 信心門檻 | L4 Bayesian Decision Theory — Reject/Doubt 機制 |
| 左右投分類 | L10 Linear Discrimination — Binary Classification |

注意：Statcast 的球種標籤本身由自動化系統產生，並非百分之百準確。部分「錯誤」預測可能反映的是標籤本身的不確定性，而非模型缺陷。

---

## 注意事項

- 所有模型均為手刻實作（numpy），不依賴 sklearn
- `spin_axis_sin/cos` 的轉換必須在 Step 2 完成，分類器不包含此轉換
- LDA 左右投分類器的特徵集（`LDA_FEATURES`）與球種分類器（`PITCH_FEATURES`）是兩組不同的特徵
- 第二層只在「前兩名預測正好是預設混淆對之一」時才觸發，其餘情況維持第一層結果
