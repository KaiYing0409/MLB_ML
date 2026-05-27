# -*- coding: utf-8 -*-
"""
Created on Sat May  9 21:57:08 2026
@author: Lin (NTHU ML Project)
"""
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

# 球種權重
# 說明：
# 'weight':權重
# 'ascending': True 代表數值越大越好；False 代表數值越小越好(如指叉球下墜)

PITCH_CONFIG = {
    # ─── 【速球系 Fastballs】 ───
    'FF': { # 四縫線速球 (Four-Seam): 追求極致的球速與向上浮力 (上竄)
        'release_speed':     {'weight': 0.4, 'ascending': True},
        'release_spin_rate': {'weight': 0.2, 'ascending': True},
        'pfx_z':             {'weight': 0.4, 'ascending': True},  # 向上竄
        'pfx_x_abs':         {'weight': 0.0, 'ascending': True}
    },
    'SI': { # 伸卡球 / 二縫線 (Sinker): 追求向打者內角竄(X)且帶有下沉(Z)
        'release_speed':     {'weight': 0.3, 'ascending': True},
        'release_spin_rate': {'weight': 0.0, 'ascending': True},  # 轉速非重點，靠縫線效應
        'pfx_z':             {'weight': 0.3, 'ascending': False}, # 越往下沉越好
        'pfx_x_abs':         {'weight': 0.4, 'ascending': True}   # 橫移越大越好
    },
    'FC': { # 卡特球 (Cutter): 像直球一樣快，但在本壘板前有微小橫向切入
        'release_speed':     {'weight': 0.4, 'ascending': True},
        'release_spin_rate': {'weight': 0.2, 'ascending': True},
        'pfx_z':             {'weight': 0.1, 'ascending': True},
        'pfx_x_abs':         {'weight': 0.3, 'ascending': True}   # 微小但銳利的橫移
    },

    # ─── 【變化球系 Breaking Balls】 ───
    'SL': { # 滑球 (Slider): 傳統滑球，速度偏快，帶有銳利的橫向與些微下墜
        'release_speed':     {'weight': 0.2, 'ascending': True},  # 丟得快的滑球很難打
        'release_spin_rate': {'weight': 0.3, 'ascending': True},
        'pfx_z':             {'weight': 0.1, 'ascending': False}, # 稍微下墜
        'pfx_x_abs':         {'weight': 0.4, 'ascending': True}   # 橫向變化
    },
    'ST': { # 橫掃球 (Sweeper): 現代棒球顯學，極端的純橫向位移
        'release_speed':     {'weight': 0.1, 'ascending': True},
        'release_spin_rate': {'weight': 0.3, 'ascending': True},
        'pfx_z':             {'weight': 0.0, 'ascending': True},  # 不看重下墜
        'pfx_x_abs':         {'weight': 0.6, 'ascending': True}   # 極致的橫移 (權重給到60%)
    },
    'CU': { # 曲球 (Curveball): 傳統大曲球，靠強烈上旋製造極端垂直掉落
        'release_speed':     {'weight': 0.1, 'ascending': True},
        'release_spin_rate': {'weight': 0.4, 'ascending': True},  # 高轉速引擎
        'pfx_z':             {'weight': 0.5, 'ascending': False}, # 極致的下墜 (越負越好)
        'pfx_x_abs':         {'weight': 0.0, 'ascending': True}
    },

    # ─── 【變速與慢速球系 Off-speed】 ───
    'CH': { # 變速球 (Changeup): 靠速差混淆，並帶有下墜與手臂側褪去(Fade)的軌跡
        'release_speed':     {'weight': 0.1, 'ascending': False}, # 越慢越能拉開與直球的速差
        'release_spin_rate': {'weight': 0.1, 'ascending': False}, # 轉速越低越能消減浮力
        'pfx_z':             {'weight': 0.4, 'ascending': False}, # 下墜
        'pfx_x_abs':         {'weight': 0.4, 'ascending': True}   # 手臂側橫移
    },
    'FS': { # 指叉球 (Splitter): 直球的軌跡，但在本壘板前因「無轉速」而夾帶重力急墜
        'release_speed':     {'weight': 0.2, 'ascending': True},  # 出手要像直球一樣快
        'release_spin_rate': {'weight': 0.3, 'ascending': False}, # 轉速越低越完美！(抹除旋轉)
        'pfx_z':             {'weight': 0.5, 'ascending': False}, # 像自由落體一樣下墜
        'pfx_x_abs':         {'weight': 0.0, 'ascending': True}
    }
}

# 第一階段：資料讀取與預處理
print("1. 正在讀取並篩選資料...")
df_raw = pd.read_csv('statcast_bat_tracking_2024_2025.csv')
target_pitches = list(PITCH_CONFIG.keys()) # 直接從矩陣抓球種名稱！

core_columns = [
    'pitch_type', 'release_speed', 'release_spin_rate', 
    'pfx_x', 'pfx_z','spin_axis','release_pos_x', 'release_pos_z', 
    'plate_x', 'plate_z'
]

df_clean = df_raw[df_raw['pitch_type'].isin(target_pitches)][core_columns].dropna().copy()
df_clean['pfx_x_abs'] = df_clean['pfx_x'].abs()
print(f"篩選與清理完成，共剩下 {len(df_clean)} 筆有效資料。\n")

# 第二階段：特徵工程 - 動態計算球種 PR 值
print("2. 正在計算各球種 PR 值 (Stuff+ Score)...")

def calculate_pitch_pr(df, config_matrix):
    df['pitch_quality_score'] = 0.0
    pitch_types = df['pitch_type'].unique()
    
    for p_type in pitch_types:
        if p_type not in config_matrix:
            continue # 如果資料庫有未知球種，直接跳過
            
        mask = df['pitch_type'] == p_type
        score = 0.0
        
        # 動態讀取矩陣中的設定來計算
        for feature, settings in config_matrix[p_type].items():
            if settings['weight'] > 0: # 權重為 0 的就不浪費時間算
                # 計算 PR 值，並直接套用 ascending 規則
                pr = df.loc[mask, feature].rank(pct=True, ascending=settings['ascending'])
                score += pr * settings['weight']
                
        df.loc[mask, 'pitch_quality_score'] = score * 100

    return df

df_scored = calculate_pitch_pr(df_clean, PITCH_CONFIG)
print("計算完成！\n")

# 特徵的最大最小區間
# 1. 指定你想檢查的特徵欄位
features_to_check = ['release_speed', 'release_spin_rate', 'pfx_z', 'pfx_x_abs']

# 2. 依照球種分組，並計算這些特徵的 max(最大值) 和 min(最小值)
# 這裡加一個 'mean' (平均值) 當作對照會更有感覺！
pitch_stats = df_clean.groupby('pitch_type')[features_to_check].agg(['max', 'min', 'mean'])

# 3. 為了讓終端機印出來比較漂亮，我們把小數點四捨五入到第 2 位
pitch_stats = pitch_stats.round(2)

print(pitch_stats)
# 第三階段：視覺化與資料匯出
print("3. 繪製圖表並匯出資料給隊友...")
avg_scores = df_scored.groupby('pitch_type')['pitch_quality_score'].mean()
print("各球種平均 PR 值 (理想應接近 50):")
print(avg_scores)

plt.figure(figsize=(8, 5))
fastballs = df_scored[df_scored['pitch_type'] == 'FF']
fastballs['pitch_quality_score'].hist(bins=30, edgecolor='black', color='skyblue')
plt.title('Distribution of FF (Four-Seam Fastball) PR Scores')
plt.xlabel('PR Score (0-100)')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()

df_scored.to_csv('ml_ready_pitch_data.csv', index=False)
print("檔案已儲存為 'ml_ready_pitch_data.csv'！\n")

# 第四階段：單顆新球評估 (動態讀取矩陣)
def evaluate_new_pitch(new_pitch, baseline_df, config_matrix):
    p_type = new_pitch['pitch_type']
    
    if p_type not in config_matrix:
        return f"錯誤：矩陣中沒有設定 {p_type} 的權重！"
        
    ref_data = baseline_df[baseline_df['pitch_type'] == p_type]
    if len(ref_data) == 0:
        return f"錯誤：基準資料庫中沒有 {p_type} 的歷史數據！"
        
    # 確保新球有算出 pfx_x_abs (為了跟歷史資料對齊)
    new_pitch['pfx_x_abs'] = abs(new_pitch['pfx_x'])
    
    score = 0.0 
    
    # 動態讀取矩陣進行推論
    for feature, settings in config_matrix[p_type].items():
        if settings['weight'] > 0:
            raw_pr = stats.percentileofscore(ref_data[feature], new_pitch[feature], kind='weak')
            
            # 如果是 ascending=False (數值越小越好)，要把 PR 分數反轉
            final_pr = raw_pr if settings['ascending'] else (100.0 - raw_pr)
            
            score += (final_pr / 100.0) * settings['weight']

    return round(score * 100, 2)


#%% 互動式輸入數據
while True:
    print("\n" + "="*40)
    print("請輸入新球的數據 (輸入 'q' 可以離開程式)：")
    
    p_type = input("球種 (例如 FF, ST, FS, SI): ").upper()
    if p_type == 'Q':
        print("系統關閉")
        break
        
    try:
        speed = float(input("球速 (mph): "))
        spin = float(input("轉速 (rpm): "))  
        pfx_z = float(input("縱向位移 (英呎): "))
        pfx_x = float(input("橫向位移 (英呎): "))
    except ValueError:
        print("格式錯誤！請輸入數字。")
        continue
        
    user_pitch = {
        'pitch_type': p_type,
        'release_speed': speed,
        'release_spin_rate': spin,
        'pfx_x': pfx_x,
        'pfx_z': pfx_z
    }    
    try:
        # 注意這裡多傳了一個參數 PITCH_CONFIG 給函數
        score = evaluate_new_pitch(user_pitch, df_clean, PITCH_CONFIG)
        print(f"\n分析完成！你的 {p_type} 綜合 PR 評分為：【 {score} 分 】")
    except Exception as e:
        print(f"計算時發生錯誤: {e}")