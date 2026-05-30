# -*- coding: utf-8 -*-
"""
Created on Sun May 10 01:12:50 2026

@author: user
"""


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# 【新增】載入你們完整的系統串接模組
from final_v1 import run_hybrid_ai_system, df_clean

# ==========================================
# 0. 網頁頁面基本設定
# ==========================================
st.set_page_config(
    page_title="AI 棒球教練：全方位投球診斷系統",
    page_icon="⚾",
    layout="wide"
)

# ==========================================
# 1. 投手設定 (因為現在的系統需要看出手點)
# ==========================================
# 我們設定一組預設的投手身體條件 (可擴充讓使用者輸入)
DEFAULT_PITCHER = {
    'release_pos_x': -2.1,
    'release_pos_z': 5.5,
    'release_extension': 6.5
}

# ==========================================
# 2. Streamlit UI 介面
# ==========================================
st.sidebar.header("📊 系統資訊")
st.sidebar.write(f"資料庫歷史球數：{len(df_clean):,} 筆")
st.sidebar.divider()
st.sidebar.markdown("""
**系統運作原理：**
1. **分類模型** 辨識未知球種。
2. **Stuff+ 系統** 評估球路威脅度。
3. **馬氏距離模型** 預測進壘成功率並給予物理修正建議。
""")

st.title("⚾ 王牌投手：全方位投球診斷系統")
st.markdown("輸入測速槍與雷達捕捉到的出手瞬間數據，AI 教練將為你進行全方位診斷。")

# --- 輸入區塊 ---
with st.container():
    st.markdown("### 1️⃣ 輸入投球原始數據")
    # 這裡我們不讓使用者選球種了，因為這是讓 AI "盲測" 的系統
    col1, col2, col3 = st.columns(3)
    
    with col1:
        speed = st.number_input("球速 Speed (mph)", value=92.0, step=0.1)
        spin = st.number_input("轉速 Spin Rate (rpm)", value=2200, step=10)
        spin_axis = st.number_input("轉軸 Spin Axis (°)", value=210.0, step=1.0)
    
    with col2:
        pfx_x = st.number_input("橫向位移 H-Break (ft)", value=-1.2, step=0.01)
        pfx_z = st.number_input("縱向位移 V-Break (ft)", value=1.5, step=0.01)
        arm_angle = st.number_input("手臂角度 Arm Angle (°)", value=28.0, step=1.0)
        
    with col3:
        target_zone = st.selectbox("🎯 目標落點區域 (Zone)", [1,2,3,4,5,6,7,8,9,11,12,13,14], index=8) # 預設選 9
        api_break_x_arm = st.number_input("手臂橫向位移 (api_x)", value=-4.2, step=0.1)
        api_break_z_grav = st.number_input("重力縱向位移 (api_z)", value=30.5, step=0.1)
        
    # 其他為了滿足模型需要，但在前台先寫死的特徵
    p_throws = st.selectbox("投手慣用手", ["R", "L"])
    vx0, vy0, ax, ay = -5.0, -135.0, -7.0, 26.0 # 預設幾何參數

# --- 執行與顯示 ---
if st.button("🚀 啟動 AI 診斷", type="primary"):
    
    # 建立要餵給系統的字典
    raw_pitch_input = {
        'release_speed': speed,
        'release_spin_rate': spin,
        'spin_axis': spin_axis,
        'api_break_x_arm': api_break_x_arm,
        'api_break_z_with_gravity': api_break_z_grav,
        'pfx_x': pfx_x,
        'pfx_z': pfx_z,
        'arm_angle': arm_angle,
        'p_throws': p_throws,
        # 模型需要的基本飛行參數
        'vx0': vx0, 'vy0': vy0, 'ax': ax, 'ay': ay
    }
    
    with st.spinner('🤖 AI 教練正在比對歷史黃金資料庫，請稍候...'):
        try:
            # 呼叫你們的主程式
            result = run_hybrid_ai_system(
                raw_pitch_data=raw_pitch_input,
                target_zone=target_zone,
                pitcher_profile=DEFAULT_PITCHER,
                df_database=df_clean
            )
            
            if result.get("status") == "success":
                st.divider()
                st.markdown("## 📋 AI 診斷報告")
                
                # 第一排：三大核心指標
                r1, r2, r3 = st.columns(3)
                r1.metric(label="🔍 AI 辨識球種", value=result.get("pitch_type"))
                
                # 這裡目前因為你的 final_v1 沒把 score 傳出來，如果有傳的話可以加上去，這裡我們先寫個範例
                r2.metric(label="🔥 Stuff+ 球威評分", value="見終端機輸出") 
                
                conf = result.get("score", 0)
                r3.metric(label=f"🎯 投進 {target_zone} 號位信心度", value=f"{conf:.1f}%")
                
                # 第二排：教練建議 (拆解 Errors)
                st.markdown("### 💡 教練修正建議 (與完美落點的物理誤差)")
                
                errors = result.get("errors", {})
                e1, e2, e3 = st.columns(3)
                
                if 'release_speed' in errors:
                    diff_speed = errors['release_speed']
                    label_s = "加速" if diff_speed < 0 else "減速"
                    e1.info(f"**球速修正**\n\n需要 {label_s} **{abs(diff_speed):.1f}** mph")
                    
                if 'release_spin_rate' in errors:
                    diff_spin = errors['release_spin_rate']
                    label_sp = "加轉" if diff_spin < 0 else "減轉"
                    e2.info(f"**轉速修正**\n\n需要 {label_sp} **{abs(diff_spin):.0f}** rpm")
                    
                if 'spin_axis' in errors:
                    diff_axis = errors['spin_axis']
                    label_a = "順時針" if diff_axis < 0 else "逆時針"
                    e3.info(f"**轉軸修正**\n\n需 {label_a} 轉 **{abs(diff_axis):.1f}** 度")

                if conf >= 60:
                    st.success("🎉 **控球優異！** 這顆球的物理狀態非常接近大聯盟成功投進該位置的黃金標準！")
                elif conf < 20:
                    st.error("⚠️ **失投警告！** 出手瞬間的數據嚴重偏離預期，請根據上方教練建議調整機制！")
                else:
                    st.warning("⚠️ **具備風險。** 雖然有可能進壘，但物理條件不夠穩定。")

            else:
                st.error(f"系統診斷失敗，資料庫中可能缺乏投進 {target_zone} 號位的該球種歷史數據。")

        except Exception as e:
            st.error(f"系統發生錯誤：{str(e)}")
