# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:27:49 2026

@author: user
"""
# -*- coding: utf-8 -*-
"""
真實賽場數據驗證腳本 
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys, os, io, re
from tabulate import tabulate

# 【修改】從 final_v1 額外載入 plot_zone_distribution 函式
from final_v1 import run_hybrid_ai_system, df_clean, plot_zone_distribution

# ==========================================================
# 0. 測試參數設定區 (開關控制台)
# ==========================================================
N_TESTS = 10 # 測試 N 顆球
TARGET_ZONE = 3 # 統一測試目標為 9 號位 (右打者外角低)

# 【新增】開關 1：抽樣模式
# 'target' = 只抽真實落點在 TARGET_ZONE 的球
# 'random' = 從全資料庫隨機抽樣
SAMPLE_MODE = 'target' 

# 【新增】開關 2：是否要在每算完一顆球時，畫出該球的 14 區熱力圖？
# (注意：設為 True 時，程式會暫停等你看完圖表並關閉，才會算下一顆)
PLOT_HEATMAP = False 

# ==========================================================
# 1. 輸出攔截器：屏蔽 print() 同時把文字記錄下來供提取
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
print("📥 正在載入真實資料庫並準備測試...")
df_real = pd.read_csv('Pitch_physical_only.csv')

# 【新增】根據 SAMPLE_MODE 決定抽樣方式
if SAMPLE_MODE == 'target':
    print(f"🎯 [模式啟動] 鎖定抽樣：只抽取真實落點在 {TARGET_ZONE} 號位的球。")
    # 先過濾出真的投進目標區的球
    pool = df_real[df_real['zone'] == TARGET_ZONE]
    if len(pool) < N_TESTS:
        print(f"⚠️ 警告：該區域只有 {len(pool)} 顆球，將使用全部球數。")
        N_TESTS = len(pool)
    test_samples = pool.sample(n=N_TESTS, random_state=42).copy()
else:
    print("🎲 [模式啟動] 隨機抽樣：從全資料庫隨機抽取球路。")
    test_samples = df_real.sample(n=N_TESTS, random_state=92).copy()

results_list = []

print(f"🚀 開始測試 {N_TESTS} 顆球，這可能需要幾十秒，請稍候...")

# ==========================================================
# 3. 執行測試 (隱藏輸出並抓取關鍵數據)
# ==========================================================
correct_count = 0

for i, (_, row) in enumerate(test_samples.iterrows()):
    true_pitch = row['pitch_type']
    raw_pitch_data = row.to_dict()
    
    current_pitcher_profile = {
        'release_pos_x': row.get('release_pos_x', -2.0),
        'release_pos_z': row.get('release_pos_z', 5.5),
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

    # 【文字探勘】從攔截下來的隱藏文字中，把你要的數字挖出來
    text_log = cp.output_text
    
    # 抓取 Margin
    margin_match = re.search(r'信心 margin:\s*([\d.]+)', text_log)
    margin_val = float(margin_match.group(1)) if margin_match else None
    
    # 抓取 PR 分數
    pr_match = re.search(r'綜合 PR 評分達 【\s*([\d.]+)\s*分\s*】', text_log)
    pr_val = float(pr_match.group(1)) if pr_match else None

    # 紀錄結果
    pred_pitch = res.get('pitch_type', 'N/A')
    is_correct = (true_pitch == pred_pitch)
    if is_correct: correct_count += 1
    
    errors = res.get('errors', {})
    
    # 將誤差塞入 results_list
    results_list.append({
        "ID": i + 1,
        "真實球種": true_pitch,
        "預測球種": pred_pitch,
        "辨識": "✅" if is_correct else "❌",
        "ML_Margin": margin_val,
        "Stuff+ PR": pr_val,
        "落點信心度(%)": round(res.get('score', 0), 1),
        "球速差距(mph)": round(errors.get('release_speed', 0), 2),
        "轉速差距(rpm)": round(errors.get('release_spin_rate', 0), 1),
        "轉軸差距(°)": round(errors.get('spin_axis', 0), 1),
        "plate_x": row.get('plate_x'), 
        "plate_z": row.get('plate_z')  
    })

    # 【新增】判斷是否要畫出單球熱力圖
    if PLOT_HEATMAP and res.get("status") == "success" and "zone_distribution" in res:
        print(f"\n📊 正在顯示第 {i+1} 顆球的區域機率分佈 (關閉視窗後將繼續執行)...")
        plot_zone_distribution(
            zone_distribution=res['zone_distribution'],
            target_zone=TARGET_ZONE,
            pitch_type=res['pitch_type'],
            target_score=res['score']
        )

# ==========================================================
# 4. 輸出總結表格 (DataFrame)
# ==========================================================
df_results = pd.DataFrame(results_list)

print("\n" + "=" * 65)
print(f"🏆 盲測結束！整體球種辨識準確率: {correct_count}/{N_TESTS} ({(correct_count/N_TESTS)*100:.1f}%)")
print("=" * 65)

# 隱藏座標，顯示前 20 筆完整數據
df_display = df_results.drop(columns=['plate_x', 'plate_z']).head(20)

# 使用 tabulate 轉成對齊好的表格字串
print(f"\n📋 【測試結果預覽 (顯示前 20 筆，共 {N_TESTS} 筆)】:")
print(tabulate(df_display, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".2f"))

# ==========================================================
# 5. 繪製好球帶九宮格與落點分佈
# ==========================================================
print("\n🎨 正在繪製這批測試球的實際落點圖...")

sz_top = 3.5
sz_bot = 1.5
sz_left = -0.83
sz_right = 0.83

fig, ax = plt.subplots(figsize=(6, 8))

rect = patches.Rectangle((sz_left, sz_bot), sz_right - sz_left, sz_top - sz_bot, 
                         linewidth=2, edgecolor='black', facecolor='none')
ax.add_patch(rect)

ax.plot([sz_left, sz_right], [sz_bot + (sz_top-sz_bot)/3, sz_bot + (sz_top-sz_bot)/3], color='gray', linestyle='--')
ax.plot([sz_left, sz_right], [sz_bot + 2*(sz_top-sz_bot)/3, sz_bot + 2*(sz_top-sz_bot)/3], color='gray', linestyle='--')
ax.plot([sz_left + (sz_right-sz_left)/3, sz_left + (sz_right-sz_left)/3], [sz_bot, sz_top], color='gray', linestyle='--')
ax.plot([sz_left + 2*(sz_right-sz_left)/3, sz_left + 2*(sz_right-sz_left)/3], [sz_bot, sz_top], color='gray', linestyle='--')

correct_pts = df_results[df_results['辨識'] == '✅']
error_pts = df_results[df_results['辨識'] == '❌']

ax.scatter(correct_pts['plate_x'], correct_pts['plate_z'], color='green', alpha=0.6, label='Correctly Identified')
ax.scatter(error_pts['plate_x'], error_pts['plate_z'], color='red', marker='x', s=60, label='Misidentified')

ax.text(0.5, 1.8, f'Zone {TARGET_ZONE}', fontsize=12, color='blue', fontweight='bold', ha='center')

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(0, 5)
# ax.invert_xaxis() #換成投手視角
ax.set_aspect('equal')
ax.set_title(f"Pitch Locations - {N_TESTS} Samples ({SAMPLE_MODE})\n(Target: Zone {TARGET_ZONE})", fontsize=14)
ax.set_xlabel("Horizontal Location (ft)")
ax.set_ylabel("Vertical Height (ft)")
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()
