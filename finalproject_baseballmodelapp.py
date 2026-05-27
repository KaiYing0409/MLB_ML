# -*- coding: utf-8 -*-
"""
Created on Sun May 10 01:12:50 2026

@author: user
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt

# ==========================================
# 0. 網頁頁面基本設定
# ==========================================
st.set_page_config(
    page_title="王牌投手：Pitch Quality 評估系統",
    page_icon="⚾",
    layout="wide"
)

# 1. 核心設定區：球種權重與邏輯矩陣 (PITCH_CONFIG)

PITCH_CONFIG = {
    'FF': { # 四縫線速球
        'release_speed':     {'weight': 0.4, 'ascending': True},
        'release_spin_rate': {'weight': 0.2, 'ascending': True},
        'pfx_z':             {'weight': 0.4, 'ascending': True},
        'pfx_x_abs':         {'weight': 0.0, 'ascending': True}
    },
    'SI': { # 伸卡球
        'release_speed':     {'weight': 0.3, 'ascending': True},
        'release_spin_rate': {'weight': 0.0, 'ascending': True},
        'pfx_z':             {'weight': 0.3, 'ascending': False},
        'pfx_x_abs':         {'weight': 0.4, 'ascending': True}
    },
    'FC': { # 卡特球
        'release_speed':     {'weight': 0.4, 'ascending': True},
        'release_spin_rate': {'weight': 0.2, 'ascending': True},
        'pfx_z':             {'weight': 0.1, 'ascending': True},
        'pfx_x_abs':         {'weight': 0.3, 'ascending': True}
    },
    'SL': { # 滑球
        'release_speed':     {'weight': 0.2, 'ascending': True},
        'release_spin_rate': {'weight': 0.3, 'ascending': True},
        'pfx_z':             {'weight': 0.1, 'ascending': False},
        'pfx_x_abs':         {'weight': 0.4, 'ascending': True}
    },
    'ST': { # 橫掃球
        'release_speed':     {'weight': 0.1, 'ascending': True},
        'release_spin_rate': {'weight': 0.3, 'ascending': True},
        'pfx_z':             {'weight': 0.0, 'ascending': True},
        'pfx_x_abs':         {'weight': 0.6, 'ascending': True}
    },
    'CU': { # 曲球
        'release_speed':     {'weight': 0.1, 'ascending': True},
        'release_spin_rate': {'weight': 0.4, 'ascending': True},
        'pfx_z':             {'weight': 0.5, 'ascending': False},
        'pfx_x_abs':         {'weight': 0.0, 'ascending': True}
    },
    'CH': { # 變速球
        'release_speed':     {'weight': 0.1, 'ascending': False},
        'release_spin_rate': {'weight': 0.1, 'ascending': False},
        'pfx_z':             {'weight': 0.4, 'ascending': False},
        'pfx_x_abs':         {'weight': 0.4, 'ascending': True}
    },
    'FS': { # 指叉球
        'release_speed':     {'weight': 0.2, 'ascending': True},
        'release_spin_rate': {'weight': 0.3, 'ascending': False},
        'pfx_z':             {'weight': 0.5, 'ascending': False},
        'pfx_x_abs':         {'weight': 0.0, 'ascending': True}
    }
}

# 2. 數據載入與快取 (處理 139 萬筆資料)
@st.cache_data
def load_and_prep_data():
    # 注意：請確保 CSV 檔案與此程式碼放在同一個資料夾，或是使用絕對路徑
    file_path = 'statcast_bat_tracking_2024_2025.csv'
    try:
        df_raw = pd.read_csv(file_path)
    except FileNotFoundError:
        # 如果找不到，嘗試使用你之前的絕對路徑 (請根據你的電腦修改)
        alt_path = r'C:\Users\user\Desktop\COLLEGE\大三後\機器學習2026\ML2026\期末project\data\statcast_bat_tracking_2024_2025.csv'
        df_raw = pd.read_csv(alt_path)
        
    target_pitches = list(PITCH_CONFIG.keys())
    core_columns = ['pitch_type', 'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z']
    
    df_clean = df_raw[df_raw['pitch_type'].isin(target_pitches)][core_columns].dropna().copy()
    df_clean['pfx_x_abs'] = df_clean['pfx_x'].abs()
    
    # 事先計算好所有球的 PR (Stuff+ 分數)，這部分也會被快取
    df_clean['pitch_quality_score'] = 0.0
    for p_type in target_pitches:
        mask = df_clean['pitch_type'] == p_type
        score = 0.0
        for feat, setts in PITCH_CONFIG[p_type].items():
            if setts['weight'] > 0:
                pr = df_clean.loc[mask, feat].rank(pct=True, ascending=setts['ascending'])
                score += pr * setts['weight']
        df_clean.loc[mask, 'pitch_quality_score'] = score * 100
        
    return df_clean

# 載入資料
with st.spinner('正在載入大聯盟實戰數據庫...'):
    df_baseline = load_and_prep_data()


# 3. 核心計算函數

def evaluate_single_pitch(new_pitch, baseline_df, config_matrix):
    p_type = new_pitch['pitch_type']
    ref_data = baseline_df[baseline_df['pitch_type'] == p_type]
    
    new_pitch['pfx_x_abs'] = abs(new_pitch['pfx_x'])
    score = 0.0 
    
    for feature, settings in config_matrix[p_type].items():
        if settings['weight'] > 0:
            raw_pr = stats.percentileofscore(ref_data[feature], new_pitch[feature], kind='weak')
            final_pr = raw_pr if settings['ascending'] else (100.0 - raw_pr)
            score += (final_pr / 100.0) * settings['weight']

    return round(score * 100, 2)

# 4. Streamlit UI 介面

st.sidebar.header("📊 數據概況")
st.sidebar.write(f"總樣本數：{len(df_baseline):,} 筆")
st.sidebar.divider()
st.sidebar.write("各球種平均 PR 值：")
st.sidebar.dataframe(df_baseline.groupby('pitch_type')['pitch_quality_score'].mean().round(2))

st.title("⚾ 王牌投手：球種 Stuff+ 評估 App")
st.markdown("透過歷史實戰數據，評估你的投球在同球種中處於什麼樣的水平。")

# --- 輸入區塊 ---
with st.container():
    p_type = st.selectbox("請選擇球種 (Pitch Type)", list(PITCH_CONFIG.keys()))
    
    col1, col2 = st.columns(2)
    with col1:
        speed = st.number_input("球速 Release Speed (mph)", value=92.0, step=0.1)
        spin = st.number_input("轉速 Spin Rate (rpm)", value=2200, step=10)
    with col2:
        pfx_x = st.number_input("橫向位移 Horizontal Break (ft)", value=0.0, step=0.01)
        pfx_z = st.number_input("縱向位移 Vertical Break (ft)", value=1.0, step=0.01)

# --- 執行與顯示 ---
if st.button("🚀 開始評估這顆球", type="primary"):
    user_pitch = {
        'pitch_type': p_type,
        'release_speed': speed,
        'release_spin_rate': spin,
        'pfx_x': pfx_x,
        'pfx_z': pfx_z
    }
    
    score = evaluate_single_pitch(user_pitch, df_baseline, PITCH_CONFIG)
    
    st.divider()
    
    # 顯示分數
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric(label="綜合評分", value=f"{score} 分")
    with c2:
        if score >= 80:
            st.balloons()
            st.success("🔥 **極品等級！** 這顆球擁有大聯盟頂尖的素質，打者很難有效擊中。")
        elif score >= 50:
            st.info("👍 **優秀水平。** 這是一顆具有實戰價值的投球，具備一定的壓制力。")
        else:
            st.warning("⚠️ **有待加強。** 這顆球的軌跡較平庸，容易被大聯盟打者鎖定進攻。")

    # 顯示圖表
    st.markdown(f"### 📈 {p_type} 歷史分布對照 (PR 分數)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ref_data = df_baseline[df_baseline['pitch_type'] == p_type]
    ax.hist(ref_data['pitch_quality_score'], bins=50, edgecolor='black', color='skyblue', alpha=0.7)
    
    # 畫出目前輸入球的位置
    ax.axvline(score, color='red', linestyle='--', linewidth=2, label=f'Your Pitch ({score})')
    ax.legend()
    ax.set_title(f'Statcast {p_type} Quality Distribution')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)