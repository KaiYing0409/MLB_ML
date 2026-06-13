# -*- coding: utf-8 -*-
"""
⚾ AI 棒球虛擬教練 - MLB-Grade Streamlit Dashboard
"""
import streamlit as st
import pandas as pd
import altair as alt
from final_v1 import run_hybrid_ai_system, df_clean

# ==========================================
# 0. 網頁基本設定 (使用寬版與暗色主題感)
# ==========================================
st.set_page_config(page_title="AI 棒球教練系統", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")

# 標題與簡介
st.title("⚾ 混合式 AI 棒球投球優化系統")
st.markdown("**National Tsing Hua University (NTHU) ESS - 運動科學專題** | *Powered by Gradient Descent & Mahalanobis Distance*")
st.markdown("---")

# ==========================================
# 1. 建立分頁 (Tabs) 讓結構更清晰
# ==========================================
tab_dashboard, tab_history, tab_theory = st.tabs(["🎯 實戰模擬診斷儀表板", "📊 盲測驗證數據庫", "🔬 演算法原理解析"])

with tab_dashboard:
    # 建立左右雙欄 (左邊輸入 30%，右邊儀表板 70%)
    col_input, col_dashboard = st.columns([3, 7])
    
    # ------------------------------------------
    # 左側：專業數據輸入面板
    # ------------------------------------------
    with col_input:
        st.markdown("### 📝 投球數據輸入 (Input)")
        with st.form("pitch_input_form"):
            st.markdown("##### 📍 戰術設定")
            target_zone = st.selectbox("預期進壘位置 (Zone 1-9)", [1, 2, 3, 4, 5, 6, 7, 8, 9], index=8)
            p_throws = st.selectbox("投手慣用手", ["R", "L"])
            
            st.markdown("##### ⚾ 核心軌跡特徵")
            release_speed = st.number_input("球速 Speed (mph)", value=92.0, step=0.5)
            release_spin_rate = st.number_input("轉速 Spin Rate (rpm)", value=1520.0, step=10.0)
            spin_axis = st.number_input("轉軸 Spin Axis (°)", value=238.0, step=1.0)
            pfx_x = st.number_input("橫向位移 HB (ft)", value=-2.3, step=0.1)
            pfx_z = st.number_input("縱向位移 VB (ft)", value=1.5, step=0.1)
            
            with st.expander("🛠️ 進階生物力學與動力參數 (Auto-filled)"):
                release_pos_x = st.number_input("出手點 X", value=-2.12)
                release_pos_z = st.number_input("出手點 Z", value=5.54)
                release_extension = st.number_input("延伸距離", value=6.5)
                arm_angle = st.number_input("出手角度", value=28.0)
                vx0 = st.number_input("初速 vx0", value=8.4)
                vy0 = st.number_input("初速 vy0", value=-135.5)
                vz0 = st.number_input("初速 vz0", value=-3.4)
                ax = st.number_input("加速度 ax", value=-6.8)
                ay = st.number_input("加速度 ay", value=26.8)
                az = st.number_input("加速度 az", value=-25.0)
                api_break_x_arm = st.number_input("api_break_x_arm", value=-4.2)
                api_break_z_with_gravity = st.number_input("api_break_z", value=30.5)

            submit_button = st.form_submit_button("🚀 執行 AI 優化診斷", use_container_width=True)

    # ------------------------------------------
    # 右側：高階數據儀表板
    # ------------------------------------------
    with col_dashboard:
        if not submit_button:
            # 尚未輸入時的佔位畫面
            st.info("👈 請在左側輸入測速儀捕捉到的投球特徵，並點擊「執行 AI 優化診斷」。")
            st.image("https://images.unsplash.com/photo-1508344928928-7105b67de451?q=80&w=1000&auto=format&fit=crop", caption="Awaiting Pitch Data...", use_container_width=True)
        
        else:
            # --- 後端運算開始 ---
            raw_pitch_input = {
                'release_speed': release_speed, 'release_spin_rate': release_spin_rate,
                'spin_axis': spin_axis, 'pfx_x': pfx_x, 'pfx_z': pfx_z,
                'release_pos_x': release_pos_x, 'release_pos_z': release_pos_z,
                'release_extension': release_extension, 'arm_angle': arm_angle,
                'p_throws': p_throws, 'vx0': vx0, 'vy0': vy0, 'vz0': vz0,
                'ax': ax, 'ay': ay, 'az': az,
                'api_break_x_arm': api_break_x_arm, 'api_break_z_with_gravity': api_break_z_with_gravity
            }
            my_pitcher = {'release_pos_x': release_pos_x, 'release_pos_z': release_pos_z, 'release_extension': release_extension}
            
            with st.spinner('🔬 正在計算多維度共變異數與梯度下降平衡點...'):
                result = run_hybrid_ai_system(raw_pitch_input, target_zone, my_pitcher, df_clean)
            
            if result.get("status") == "success":
                st.markdown("### 📊 AI 診斷分析報告 (Diagnostic Overview)")
                
                # --- Top Metrics (核心指標) ---
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("🤖 AI 辨識球種", result['pitch_type'])
                col_m2.metric("🔥 Stuff+ PR (純物理球威)", f"{result.get('stuff_score', 0.0):.1f}")
                col_m3.metric(f"📍 目標 Zone {target_zone} 信心度", f"{result.get('score', 0.0):.1f}%")
                
                st.markdown("---")
                
                # --- Middle Section (圖表與處方) ---
                col_chart, col_advice = st.columns([4, 6])
                
                with col_chart:
                    st.markdown("##### ⚾ 球路位移特徵 (Movement Profile)")
                    # 畫一張專業的位移分佈圖 (X: 橫向位移, Y: 縱向位移)
                    chart_data = pd.DataFrame({'橫向位移 (ft)': [pfx_x], '縱向位移 (ft)': [pfx_z], '球種': [result['pitch_type']]})
                    scatter = alt.Chart(chart_data).mark_circle(size=200, color='#FF4B4B').encode(
                        x=alt.X('橫向位移 (ft):Q', scale=alt.Scale(domain=[-3, 3])),
                        y=alt.Y('縱向位移 (ft):Q', scale=alt.Scale(domain=[-3, 3])),
                        tooltip=['球種', '橫向位移 (ft)', '縱向位移 (ft)']
                    ).properties(height=250)
                    
                    # 畫十字線
                    rule_x = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='gray', strokeDash=[5,5]).encode(x='x:Q')
                    rule_y = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', strokeDash=[5,5]).encode(y='y:Q')
                    
                    st.altair_chart(scatter + rule_x + rule_y, use_container_width=True)

                with col_advice:
                    advice = result.get('coach_advice', {})
                    final_conf = result.get('coach_confidence', 0.0)
                    
                    st.markdown("##### 👨‍🏫 AI 虛擬教練處方箋 (Prescription)")
                    st.success(f"⚡ 透過以下微調，預期信心度可攀升至 **{final_conf:.1f}%**")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("建議球速 (mph)", 
                              f"{release_speed + advice.get('release_speed', 0):.1f}", 
                              f"{advice.get('release_speed', 0):+.2f} mph", delta_color="inverse") # 球速降通常用 inverse 避免被當成變差
                    c2.metric("建議轉速 (rpm)", 
                              f"{release_spin_rate + advice.get('release_spin_rate', 0):.0f}", 
                              f"{advice.get('release_spin_rate', 0):+.0f} rpm")
                    c3.metric("建議轉軸 (°)", 
                              f"{spin_axis + advice.get('spin_axis', 0):.1f}", 
                              f"{advice.get('spin_axis', 0):+.1f} °")
                    
                    # --- 動態教練語錄生成器 ---
                    st.markdown("###### 🗣️ 教練白話指引：")
                    cues = []
                    if abs(advice.get('release_speed', 0)) > 0.5:
                        cues.append("手臂發力稍微放鬆，不需要刻意催球速。")
                    if abs(advice.get('spin_axis', 0)) > 2.0:
                        cues.append("放球瞬間食指微調扣縫線的角度，將轉軸稍微偏轉，這能大幅提升進壘率！")
                    if advice.get('release_spin_rate', 0) > 50.0:
                        cues.append("手指最後的延伸與下壓可以再多吃一點力量，增加旋轉力道。")
                        
                    if not cues:
                        st.info("「這球的投球機制已經非常完美了！記住現在的肌肉感覺，繼續保持！」")
                    else:
                        st.info("「" + " ".join(cues) + "」")
            else:
                st.error(f"系統分析發生錯誤：{result.get('message')}")

# ==========================================
# Tab 2 & 3: 報告用的擴充區
# ==========================================
with tab_history:
    st.markdown("### 📊 歷史大數據盲測驗證結果")
    st.write("此處可放置 `test_real_data.py` 輸出的 30 顆球盲測 DataFrame 以及 Matplotlib 九宮格進壘圖，作為系統準確度的證據。")

with tab_theory:
    st.markdown("### 🔬 演算法核心數學模型")
    st.latex(r"W_k = \left( \frac{\sigma_k}{\text{Tolerance}_k} \right)^2")
    st.latex(r"\text{Total Loss} = \text{Loss}_{\text{precision}} + (\text{Loss}_{\text{penalty}} \times \lambda)")
    st.write("展示**生理容忍度自平衡梯度下降模型 (Penalized Optimization)** 的推導過程與量綱消除魔法。")
