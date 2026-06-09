# 棒球球種分類器（Baseball Pitch Type Classifier）

利用 MLB Statcast 投球物理數據，透過階層式分類架構（QDA + BinaryLDA）對球種進行自動分類。

# MLB 投球球種分類器 — 架構與使用說明

## 概述

本分類器使用 MLB Statcast 系統提供的投球物理量測數據，對 8 種球種進行分類。分類器採用**階層式 QDA（Quadratic Discriminant Analysis）架構**，所有模型皆以 NumPy 手刻實作，不使用 sklearn。

### 分類目標（8 種球種）

| 代碼 | 球種名稱 | 大類 |
|------|---------|------|
| FF | 四縫線速球 Four-Seam Fastball | Fastball |
| SI | 伸卡球 Sinker | Fastball |
| FC | 卡特球 Cutter | Fastball |
| SL | 滑球 Slider | Breaking |
| ST | 橫掃滑球 Sweeper | Breaking |
| CU | 曲球 Curveball | Breaking |
| CH | 變速球 Changeup | Offspeed |
| FS | 分叉球 Splitter | Offspeed |

> KC（彈指曲球 Knuckle Curve）已合併至 CU。詳見 `KC_CU_合併理由.md`。

---

## 架構

```
輸入（單筆投球物理數據 + 投手慣用手 p_throws）
    │
    ├── p_throws = R ──┐
    └── p_throws = L ──┤
                       │
              ┌────────▼────────┐
              │  Layer 1 QDA    │
              │ （三大類分類）    │
              │  特徵：9 個      │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌─────────┐  ┌──────────┐  ┌──────────┐
    │Fastball │  │ Breaking │  │ Offspeed │
    │QDA (3特徵)│ │QDA (9特徵)│ │          │
    │FF/SI/FC │  │SL/ST/CU  │  │ R: QDA   │
    └─────────┘  └──────────┘  │   (8特徵) │
                               │   CH/FS  │
                               │ L: → CH  │
                               └──────────┘
```

### Layer 1：三大類 QDA

將投球分為 Fastball / Breaking / Offspeed 三大類。左右投各一個獨立的 QDA 模型。

### Layer 2：子分類器

根據 Layer 1 的預測結果，routing 至對應的子分類器：

| 子分類器 | 球種 | 說明 |
|---------|------|------|
| Fastball R/L | FF, SI, FC | 左右投共用同一組特徵，各自獨立訓練 |
| Breaking R/L | SL, ST, CU | 同上 |
| Offspeed R | CH, FS | 僅右投建模 |
| Offspeed L | — | 左投 FS 樣本極少（約 2 筆），直接輸出 CH |

---

## 特徵

### 選用原則

僅使用**球本身飛行過程中的物理量**，排除以下類型的特徵：

| 排除類型 | 排除欄位 | 理由 |
|---------|---------|------|
| 投手身體特徵 | `release_pos_x/z/y`, `release_extension`, `arm_angle` | 反映投手個人投球機制，非球的物理特性。不同投手投同一球種出手點差異大，納入會過擬合特定投手習慣。 |
| 投球結果 | `plate_x`, `plate_z`, `zone` | 球到壘板的位置是投球結果，不是物理輸入。 |
| 打者相關 | `api_break_x_batter_in`, `sz_top`, `sz_bot`, `stand` | 含打者視角偏差或打者身高資訊。 |
| 高度冗餘 | `effective_speed`, `vy0`, `ax`, `az`, `pfx_z` | 與保留特徵的 Pearson 相關係數 |r| > 0.9，攜帶相同資訊。 |

### 最終特徵清單（9 個）

| 特徵 | 說明 | 單位 |
|------|------|------|
| `release_speed` | 出手球速 | mph |
| `release_spin_rate` | 轉速 | rpm |
| `spin_axis_sin` | sin(旋轉軸角度) | — |
| `spin_axis_cos` | cos(旋轉軸角度) | — |
| `pfx_x` | 水平位移（去除重力） | 英吋 |
| `api_break_z_with_gravity` | 垂直位移（含重力） | 英吋 |
| `vx0` | 出手水平初速 | ft/s |
| `vz0` | 出手垂直初速 | ft/s |
| `ay` | 縱向加速度 | ft/s² |

> `spin_axis`（0–360°）為循環變數，直接使用會有 0°/360° 邊界問題（CU 右投分布跨越邊界，邊界樣本佔 6.4%）。因此分解為 sin/cos 兩個線性特徵。

### 各層特徵配置

**Layer 1（三大類，9 個特徵）：**
`spin_axis_sin`, `api_break_z_with_gravity`, `release_speed`, `pfx_x`, `spin_axis_cos`, `release_spin_rate`, `ay`, `vx0`, `vz0`

**Fastball 子分類器（3 個特徵）：**
`pfx_x`, `api_break_z_with_gravity`, `spin_axis_sin`

**Breaking 子分類器（9 個特徵，全部）：**
`api_break_z_with_gravity`, `pfx_x`, `release_speed`, `spin_axis_cos`, `vz0`, `spin_axis_sin`, `release_spin_rate`, `vx0`, `ay`

**Offspeed 子分類器（8 個特徵，右投專用）：**
`release_spin_rate`, `pfx_x`, `api_break_z_with_gravity`, `vx0`, `spin_axis_sin`, `spin_axis_cos`, `release_speed`, `vz0`

> 特徵數量由 Sequential Forward Selection 決定：依 F-ratio 排序逐步加入特徵，在驗證集上找準確率不再顯著提升的 elbow 點。

---

## 特徵選擇流程

```
所有物理特徵（22 個）
    │
    ▼
排除非球體物理特徵（投手身體、投球結果、打者相關）
    │
    ▼
剩餘 14 個特徵
    │
    ▼
相關係數篩選（左右投分開計算，|r| > 0.9 取聯集移除）
    │
    ▼
剩餘 9 個特徵
    │
    ▼
F-ratio 排序（左右投分開，按三大類 / 各大類內部分別計算）
    │
    ▼
Sequential Forward Selection（依 F-ratio 排序逐步加入，驗證集找 elbow）
    │
    ▼
各層 / 各大類的最終特徵清單
```

---

## 資料前處理

### 流程（step2_preprocess.py）

```
statcast_bat_tracking_2024_2025.csv（原始資料）
    │
    ▼ 去除 pitch_type 缺值
    ▼ 過濾佔比 < 1% 的球種（決定分類目標）
    ▼ KC → CU 合併（第一版分類器迭代後的決策）
    ▼ IQR 離群移除（按 pitch_type × p_throws 分組，k=3）
    │   └── spin_axis：循環感知 IQR（平移至 180° 後計算，避免邊界誤刪）
    │   └── 其他特徵：標準 IQR
    ▼ 移除模型特徵欄位缺值
    ▼ spin_axis → sin/cos 轉換
    ▼ 分層抽樣 10 萬筆（seed=42）
    │
    ▼
testdata_only_phy.csv
```

### 循環感知 IQR

`spin_axis` 為 0–360° 的循環變數，傳統 IQR 在邊界附近會誤判。處理方式：

1. 用 bin=10° 的 histogram 找該（球種 × 手性）子群的眾數角度
2. 平移使眾數落在 180°（遠離邊界）
3. 在平移空間做標準 IQR（k=3）
4. 從原始資料移除對應離群樣本（保留原始角度值）

---

## 評估指標

**Macro Accuracy**：各球種準確率的算術平均。

$$\text{Macro Accuracy} = \frac{1}{K} \sum_{k=1}^{K} \frac{\text{正確預測的第 } k \text{ 類樣本數}}{\text{第 } k \text{ 類的總樣本數}}$$

### 效能（測試集 Macro 78.8%）

| 球種 | 準確率 | 備註 |
|------|-------|------|
| FF | 94.1% | |
| SI | 84.5% | |
| FC | 72.2% | 與 SL 互相混淆 |
| SL | 63.4% | 與 FC/ST 互相混淆 |
| ST | 87.3% | |
| CU | 88.8% | |
| CH | 91.0% | |
| FS | 49.1% | 與 CH 物理高度相似，物理上限問題 |

### 已知限制

- **FC/SL 混淆**：卡特球和滑球在水平位移和球速維度上有重疊區間，屬於棒球界公認的分類難題。
- **CH/FS 混淆**：變速球和分叉球的飛行物理幾乎相同，差異主要在握球方式（Statcast 無法量測），74% 的準確率反映物理上限。
- **左投 Offspeed**：左投 FS 在訓練集僅約 2 筆，無法建模，直接輸出 CH。

---

## 資料切分

全資料一次切分（seed=42）：

| 集合 | 比例 | 用途 |
|------|------|------|
| 訓練集 | 60% | 訓練所有 QDA 模型、計算 z-score 參數 |
| 驗證集 | 20% | Forward Selection 的 elbow 判斷 |
| 測試集 | 20% | 最終評估（僅跑一次） |

---

## 檔案結構

| 檔案 | 用途 |
|------|------|
| `step1_eda.py` | EDA：球種統計、特徵分布、spin_axis 邊界分析 |
| `step2_preprocess.py` | 前處理：清理、IQR、sin/cos 轉換、抽樣 → 輸出 `testdata_only_phy.csv` |
| `step3_feature_selection.py` | 特徵選擇：相關係數篩選、F-ratio 排序 |
| `step4_forward_selection.py` | Forward Selection：找各層 / 各大類的 TOP_N |
| `step5_pipeline.py` | 完整 pipeline 端對端評估（驗證集 + 測試集） |
| `train_and_save.py` | 訓練模型並存成 `model.pkl` |
| `predict.py` | 單筆預測介面（terminal 互動 / import 使用） |

---

## 使用方式

### 1. 前處理 → 訓練 → 預測

```bash
python step2_preprocess.py      # 產出 testdata_only_phy.csv
python train_and_save.py        # 產出 model.pkl
python predict.py               # Terminal 互動預測
```

### 2. 從其他程式呼叫

```python
import predict
predict.ensure_model_loaded()

result = predict.predict_pitch({
    'p_throws': 'R',
    'release_speed': 92.0,
    'release_spin_rate': 2300.0,
    'spin_axis': 210.0,
    'pfx_x': -5.2,
    'api_break_z_with_gravity': 30.5,
    'vx0': 8.4,
    'vz0': -3.4,
    'ay': 26.8,
})

print(result['predicted_pitch'])  # e.g. 'SL'
print(result['margin'])           # Layer 1 信心分數
```

### 輸出格式

```python
{
    'predicted_pitch': 'SL',     # 預測球種代碼
    'margin':          0.8523,   # Layer 1 後驗機率差（top1 - top2）
    'hand':            'R',      # 投手慣用手
    'layer2':          False,    # 保留欄位（新架構不使用）
    'top2_candidate':  '',       # 保留欄位（新架構不使用）
}
```
---

## ⚾ 球種素質評估模組（Stuff+ PR 評分系統）

**負責人：** LinWeiLun

### 包含檔案

| 檔案 | 說明 |
|------|------|
| `finalproject_baseballmodel.py` | 核心邏輯：資料處理、PR 百分位計算、終端互動評估 |
| `finalproject_baseballmodelapp.py` | Streamlit Web App 視覺化介面 |

### 評分原理

對照 139 萬筆 Statcast 2024–2025 實戰數據，將每顆球的四項物理特徵換算成百分位排名（PR），再依球種加權求和，輸出 **0–100 的素質評分（Stuff+ Score）**。

| 分數 | 等級 |
|------|------|
| 80–100 | 🔥 極品：大聯盟頂尖素質 |
| 50–79 | 👍 優秀：具實戰壓制力 |
| 0–49 | ⚠️ 待加強：軌跡平庸易被攻略 |

### 球種權重矩陣

`↑` 數值越大越好　`↓` 數值越小越好

| 球種 | 球速 | 轉速 | 縱向位移 pfx_z | 橫向位移 |
|------|------|------|--------------|------|
| FF 四縫線速球 | 0.4 ↑ | 0.2 ↑ | 0.4 ↑（向上浮力）| — |
| SI 伸卡球 | 0.3 ↑ | — | 0.3 ↓（下沉）| 0.4 ↑ |
| FC 卡特球 | 0.4 ↑ | 0.2 ↑ | 0.1 ↑ | 0.3 ↑（銳利切入）|
| SL 滑球 | 0.2 ↑ | 0.3 ↑ | 0.1 ↓ | 0.4 ↑ |
| ST 橫掃球 | 0.1 ↑ | 0.3 ↑ | — | **0.6 ↑**（極致橫移）|
| CU 曲球 | 0.1 ↑ | 0.4 ↑ | **0.5 ↓**（極致下墜）| — |
| CH 變速球 | 0.1 ↓（拉開速差）| 0.1 ↓ | 0.4 ↓ | 0.4 ↑ |
| FS 指叉球 | 0.2 ↑ | 0.3 ↓（抹除旋轉）| **0.5 ↓**（急墜）| — |

### 執行方式

```bash
# 終端機互動版
python finalproject_baseballmodel.py

# Web App 版（推薦）
streamlit run finalproject_baseballmodelapp.py
```

### 額外套件需求

```bash
pip install streamlit scipy
```

# ⚾ MLB 混合式 AI 投球分析與優化系統 (Hybrid Pitch Analytics System)

## 📖 專案簡介 (Introduction)
本專案旨在打造一套媲美 MLB 職業球隊數據部門的 **「端到端 (End-to-End) 投球分析系統」**。
本系統接收來自雷達測速槍的單顆球原始物理數據（球速、轉速、位移等）後，會透過不同的獨立計算引擎，自動辨識球種，並平行輸出這顆球的「球威品質」與「落點估算信心值」，提供教練與選手最科學的數據回饋。

---

## 🏗️ 系統總架構：Y 字型平行運算管線 (Y-Pipeline)

本系統的資料流採用 **Y 字型分流架構**，當一顆全新的投球數據進入系統時，執行流程如下：

1. **[起點]** 接收 12 項投球原始物理特徵。
2. **[辨識]** 資料進入 **模組 A (球種辨識引擎)**，系統自動判定這顆球的身分（例如：滑球 SL）。
3. **[分流]** 系統將球種標籤與原始資料，同時派發給兩個平行運算的評分部門：
   - ➡️ **模組 B (球威品質引擎)**：專注評估軌跡的噁心程度。
   - ➡️ **模組 C (馬氏靶心引擎)**：專注評估投進目標位置的機率與物理誤差。
4. **[終點]** 匯整兩大部門的分數，產出綜合分析。

---

## 🧩 核心模組功能與底層邏輯解析

### 🟢 模組 A：AI 球種辨識引擎 (Pitch Classification)
* **功能目的：** 在不依賴人為標籤的情況下，精準判斷投手剛投出的球種。
* **計算邏輯 ：**
  本模組基於統計學手刻的 **QDA (二次判別分析)** 與 **階層式 Binary LDA (線性判別分析)**。
  - **基準線對齊：** 系統會先將球速、位移等特徵，扣除全聯盟的「四縫線速球 (FF) 基準線」，轉換為「相對特徵」，以消除不同投手基礎條件的誤差。
  - **QDA 分類：** 利用歷史數據為每個球種建立專屬的共變異數矩陣（捕捉特徵間的關聯，例如球速越快通常轉速越高）。當新球進入時，計算其在多維空間中落入各球種機率分佈的「後驗機率」，機率最高者即為預測球種。
  - **第二層防線：** 若模型對前兩名球種的判斷信心度 (Margin) 過低，會自動觸發專屬的 Binary LDA 進行一對一的死鬥重判（例如滑球 vs 卡特球），極大化辨識準確率。

### 🟢 模組 B：Stuff+ 球威品質評分系統 (Pitch Quality Engine)
* **功能目的：** 對球的數據進行物理評估員，根據球的位移軌跡與其他參數，評估這顆球的球威，滿分為 PR 100。
* **計算邏輯 (動態權重百分位數)：**
  每一種球路都有其追求的極致物理目標（例如：四縫線速球追求「上竄位移與球速」，而指叉球追求「極致的下墜與消除轉速」）。
  - **特徵權重矩陣：** 模組內建一套 MLB 級別的球種權重矩陣 (`PITCH_CONFIG`)。例如設定滑球的水平位移佔 40% 分數，球速佔 20%。
  - **PR 值映射：** 系統會去歷史資料庫 (`df_database`) 中撈出該球種的所有歷史數據，並將新球的特徵與之對比，計算出百分等級 (Percentile Rank)。最後依據權重加總，得出一個 `0 ~ 100` 的球威綜合分數。分數越高，代表這顆球的軌跡在聯盟中越難被打者擊中。

### 🟢 模組 C：Command+ 馬氏靶心控球引擎 (Target Accuracy Engine)
* **功能目的：** ，評估這顆球是否能落入預期落點，並分析可能讓球投偏的成因。
* **計算邏輯 (動態黃金標準與馬氏距離)：**
  - **Phase 1 (建立基準池)：** 系統接收到模組 A 判定的球種與教練的目標區域（如 9 號位）後，會從資料庫篩選出「成功落入該區域」且「投手身體條件（出手點、延伸距離）與當前投手最相似」的前 100 顆歷史投球，作為黃金基準池。
  - **Phase 2 (萃取完美靶心)：** 計算這 100 顆黃金球在核心特徵（如球速、轉速、轉軸）上的「平均值（完美靶心 $\mu$）」與「共變異數矩陣 ($\Sigma$)」。
  - **Phase 3 (馬氏距離評分)：** 利用馬氏距離 (Mahalanobis Distance) 計算新球與完美靶心的差異。
    $$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$
    相較於一般的直線距離，馬氏距離能透過共變異數矩陣的反矩陣 ($\Sigma^{-1}$) 消除特徵之間的物理連動副作用（例如球速變快導致轉速自然提升的合理誤差）。最終透過常態分佈衰減曲線，將距離轉換為 `0% ~ 100%` 的進壘信心度。

---

## 🚀 系統執行範例 (Execution Example)

當我們呼叫主程式 `run_hybrid_ai_system()` 並傳入一顆未知投球的 12 項測速槍數據時，終端機輸出結果如下：

```text
==================================================
⚾ [系統啟動] 接收到全新測速槍數據，開始解析...
==================================================
🤖 AI 辨識判定：這是一顆 【SL】 (信心 margin: 0.8523)

🔥 [分支 1] 評估球路軌跡品質 (Stuff+ Score)...
✅ 球威評分完成：綜合 PR 評分達 【 92.5 分 】

🧠 [分支 2] 啟動控球優化引擎 (目標落點：9 號位)...
✅ 控球評分完成：落入目標區域的信心度為 【 15.2% 】

==================================================
📊 系統終極輸出報告：
球種: SL (滑球)
球威 (Stuff+): 92.5 (極佳的物理軌跡)
控球 (Command+): 15.2% (嚴重失投)
---
