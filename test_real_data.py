# -*- coding: utf-8 -*-
"""
真實賽場數據驗證腳本 (AI 教練處方箋升級版)
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys, os, io, re
from tabulate import tabulate

# 從 final_v1 載入（請確保 final_v1 頂層的資料讀取 print 已清空）
from final_v1 import run_hybrid_ai_system, df_clean

# 0. 測試參數設定區 (開關控制台)
N_TESTS = 100      # 測試 N 顆球
TARGET_ZONE = 9    # 統一測試目標落點

# 抽樣模式：
# 'target' = 只抽真實落點在 TARGET_ZONE 的球
# 'random' = 從全資料庫隨機抽樣
SAMPLE_MODE = 'random'

# df_results 為輸出結果
# ==========================================================
# 1. 輸出攔截器：屏蔽內部 print() 避免洗版
# ==========================================================
class CapturedPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self.captured_output = io.StringIO()
        sys.stdout = self.captured_output
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        self.output_text = self.captured_output.getvalue()

# ==========================================================
# 2. 測試設定與資料載入
# ==========================================================
print("📥 正在載入真實資料庫並準備進行 AI 盲測...")
df_real = pd.read_csv('Pitch_physical_only.csv')

# 根據 SAMPLE_MODE 決定抽樣方式
if SAMPLE_MODE == 'target':
    print(f"🎯 [模式啟動] 鎖定抽樣：只抽取真實落點在 {TARGET_ZONE} 號位的球。")
    pool = df_real[df_real['zone'] == TARGET_ZONE]
    if len(pool) < N_TESTS:
        print(f"⚠️ 警告：該區域只有 {len(pool)} 顆球，將使用全部球數。")
        N_TESTS = len(pool)
    test_samples = pool.sample(n=N_TESTS, random_state=42).copy()
else:
    print("🎲 [模式啟動] 隨機抽樣：從全資料庫隨機抽取球路。")
    test_samples = df_real.sample(n=N_TESTS, random_state=92).copy()

results_list = []
print(f"🚀 開始對 {N_TESTS} 顆球進行 Pipeline 診斷與梯度下降優化，請稍候...")

# ==========================================================
# 3. 執行測試 (隱藏輸出並抓取關鍵數據)
# ==========================================================
correct_count = 0

for i, (_, row) in enumerate(test_samples.iterrows()):
    true_pitch = row['pitch_type']
    raw_pitch_data = row.to_dict()
    
    current_pitcher_profile = {
        'release_pos_x':    row.get('release_pos_x', -2.0),
        'release_pos_z':    row.get('release_pos_z', 5.5),
        'release_extension': row.get('release_extension', 6.0)
    }
    
    # 使用攔截器執行
    with CapturedPrints() as cp:
        try:
            res = run_hybrid_ai_system(
                raw_pitch_data=raw_pitch_data,
                target_zone=TARGET_ZONE, 
                pitcher_profile=current_pitcher_profile, 
                df_database=df_clean
            )
        except Exception as e:
            res = {"status": "error", "pitch_type": "Error", "score": 0.0}

    # 使用 Regex 補抓部分未納入字典的舊欄位 (如模型信心 margin)
    text_log = cp.output_text
    margin_match = re.search(r'信心 margin:\s*([\d.]+)', text_log)
    margin_val = float(margin_match.group(1)) if margin_match else 0.0

    # 檢查預測正確性
    pred_pitch = res.get('pitch_type', 'N/A')
    is_correct = (true_pitch == pred_pitch)
    if is_correct:
        correct_count += 1
    
    # 提取 Phase 3 原始物理誤差 (差距 = 實際 - 完美靶心)
    errors = res.get('errors', {})
    
    # 🌟 【本次升級核心】 提取 Phase 4 梯度下降教練建議與優化信心度
    advice = res.get('coach_advice', {})
    coach_conf = res.get('coach_confidence', 0.0)
    
    results_list.append({
        "ID":             i + 1,
        "真實球種":       true_pitch,
        "預測球種":       pred_pitch,
        "辨識":           "✅" if is_correct else "❌",
        "ML_Margin":      margin_val,
        "Stuff+ PR":      round(res.get('stuff_score', 0.0), 2),
        "目前信心(%)":    round(res.get('score', 0.0), 1),
        
        # 原始誤差 (診斷)
        "球速差距(mph)":  round(errors.get('release_speed', 0.0), 2),
        "轉速差距(rpm)":  round(errors.get('release_spin_rate', 0.0), 1),
        "轉軸差距(°)":    round(errors.get('spin_axis', 0.0), 1),
        
        # 🌟 教練處方箋 (修正量)
        "建議修正球速":   round(advice.get('release_speed', 0.0), 2),
        "建議修正轉速":   round(advice.get('release_spin_rate', 0.0), 1),
        "建議修正轉軸":   round(advice.get('spin_axis', 0.0), 1),
        "修正後信心(%)":  round(coach_conf, 1),
        
        # 繪圖用進壘點
        "plate_x":        row.get('plate_x'),
        "plate_z":        row.get('plate_z')
    })

# ==========================================================
# 4. 輸出總結表格 (精心編排防跑版欄位)
# ==========================================================
df_results = pd.DataFrame(results_list)

print("\n" + "=" * 75)
print(f"🏆 盲測與優化結束！整體球種辨識準確率: {correct_count}/{N_TESTS} ({(correct_count/N_TESTS)*100:.1f}%)")
print("=" * 75)

# 為了防止全欄位印出導致終端機嚴重換行換到崩潰，我們篩選核心欄位展示
columns_to_show = [
    "ID", "真實球種", "預測球種", "辨識", "Stuff+ PR", 
    "目前信心(%)", "建議修正球速", "建議修正轉速", "建議修正轉軸", "修正後信心(%)"
]
df_display = df_results[columns_to_show].head(20)

print(f"\n📋 【AI 診斷與處方箋盲測預覽 (顯示前 20 筆，共 {N_TESTS} 筆)】:")
print(tabulate(df_display, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".2f"))

# ==========================================================
# 5. 繪製好球帶九宮格與落點分佈
# ==========================================================
print("\n🎨 正在繪製這批測試球的實際落點分佈圖...")

sz_top   =  3.5
sz_bot   =  1.5
sz_left  = -0.83
sz_right =  0.83

fig, ax = plt.subplots(figsize=(6, 8))

# 繪製主好球帶外框
rect = patches.Rectangle(
    (sz_left, sz_bot), sz_right - sz_left, sz_top - sz_bot,
    linewidth=2, edgecolor='black', facecolor='none'
)
ax.add_patch(rect)

# 繪製九宮格內部虛線
for frac in [1/3, 2/3]:
    ax.plot([sz_left, sz_right],
            [sz_bot + frac*(sz_top-sz_bot), sz_bot + frac*(sz_top-sz_bot)],
            color='gray', linestyle='--')
    ax.plot([sz_left + frac*(sz_right-sz_left), sz_left + frac*(sz_right-sz_left)],
            [sz_bot, sz_top],
            color='gray', linestyle='--')

correct_pts = df_results[df_results['辨識'] == '✅']
error_pts   = df_results[df_results['辨識'] == '❌']

# 點出預測正確與失敗的進壘點
ax.scatter(correct_pts['plate_x'], correct_pts['plate_z'],
           color='green', alpha=0.6, label='Correctly Identified')
ax.scatter(error_pts['plate_x'],   error_pts['plate_z'],
           color='red', marker='x', s=60, label='Misidentified')

# 標註當前鎖定的測試區域
ax.text(0.5, 1.8, f'Zone {TARGET_ZONE}', fontsize=12,
        color='blue', fontweight='bold', ha='center')

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(0, 5)
ax.set_aspect('equal')
ax.set_title(f"Pitch Locations - {N_TESTS} Samples ({SAMPLE_MODE})\n(Target: Zone {TARGET_ZONE})", fontsize=14)
ax.set_xlabel("Horizontal Location (ft)")
ax.set_ylabel("Vertical Height (ft)")
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()
