import pandas as pd
import numpy as np
import predict
# 載入球種預測模型 
from predict import predict_pitch
from pathlib import Path
from scipy import stats
# 載入品質分析模型 
from finalproject_baseballmodel import evaluate_new_pitch, PITCH_CONFIG
# ==========================================
predict.ensure_model_loaded()
print('MODEL_AVAILABLE =', predict.MODEL_AVAILABLE)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Pitch_physical_only.csv"
print(f"正在讀取資料: {DATA_PATH} ...")

df = pd.read_csv(DATA_PATH)

# 2. 定義核心欄位並篩選 
columns_to_keep = [
    'pitch_type',         # 球種
    'release_pos_x',      # 出手點橫向位置
    'release_pos_z',      # 出手高度
    'release_extension',  # 延伸距離
    'zone',               # 實際落點區域 (1~14)
    'release_speed',      # 球速
    'release_spin_rate',  # 轉速
    'spin_axis',          # 轉軸
    'pfx_x',              # 水平位移量
    'pfx_z'               # 垂直位移量
]
df_clean = df[columns_to_keep].copy()
df_clean['pfx_x_abs'] = df_clean['pfx_x'].abs()

#%%
# 預覽前五筆資料
print(df_clean.head())

#phase 1 : 建立「成功落點」+「身體條件相似」的黃金基準池
def build_success_similarity_pool(df, target_pitch_type, target_zone, target_pitcher, top_k=100):
    """
    先鎖定「成功落點」，再找「身體條件最相似」的投手，建立基準池。
    
    參數:
    df: 清理過後的 Statcast DataFrame 
    target_pitch_type: 目標球種 (如 'SL')
    target_zone: 預期落點網格 (如 9)
    target_pitcher: 目標投手的物理特徵字典
    top_k: 要萃取多少顆黃金標準球
    """
    
    physical_features = ['release_pos_x', 'release_pos_z', 'release_extension']
    
    print(f"stage 1 ：尋找所有落入 {target_zone} 號位的 {target_pitch_type}...")
    
    # 1. 鎖定「該球種」且「成功落入目標區域」的球
    df_success = df[(df['pitch_type'] == target_pitch_type) & (df['zone'] == target_zone)].copy()
    
    # 剔除有缺失值的資料
    df_success = df_success.dropna(subset=physical_features)
    
    if len(df_success) == 0:
        raise ValueError(f"錯誤：資料庫中找不到落入 {target_zone} 號位的 {target_pitch_type} 數據！")
    
    if len(df_success) < top_k:
        print(f"警告：符合條件的總球數 ({len(df_success)}) 少於設定的 top_k ({top_k})。將取用全部符合的數據。")
        top_k = len(df_success)
    else:
        print(f"過濾完成：共有 {len(df_success)} 顆成功的球。")

    print(f"stage 2 ：在這群成功案例中，尋找與目標投手最相似的 Top {top_k} 顆球...")
    
    # 2. 提取矩陣準備算距離
    X_database = df_success[physical_features].values
    X_target = np.array([
        target_pitcher['release_pos_x'], 
        target_pitcher['release_pos_z'], 
        target_pitcher['release_extension']
    ])
    
    # Z-score 標準化 
    mu_phys = np.mean(X_database, axis=0)
    sigma_phys = np.std(X_database, axis=0)
    sigma_phys[sigma_phys == 0] = 1e-8 # 
    
    X_database_scaled = (X_database - mu_phys) / sigma_phys
    X_target_scaled = (X_target - mu_phys) / sigma_phys
    
    # 歐式距離計算
    distances = np.linalg.norm(X_database_scaled - X_target_scaled, axis=1)
    
    # 排序並找出距離最短的前 K 名的索引
    nearest_indices = np.argsort(distances)[:top_k]
    
    # 3. 建立並回傳最終的「黃金基準池」
    gold_pool_df = df_success.iloc[nearest_indices].copy()
    gold_pool_df['physical_distance'] = distances[nearest_indices]
    
    print("基準池建立完成！")
    print("-" * 30)
    
    return gold_pool_df


def run_phase2_analysis(pool_df, optimization_features=None):
    """Phase 2：計算優化目標特徵的平均值與共變異數矩陣。"""
    if optimization_features is None:
        optimization_features = ['release_speed', 'release_spin_rate', 'spin_axis']

    print("\nphase 2 ：進入優化分析階段...")
    print(f"使用的優化特徵: {optimization_features}")

    pool_df_clean = pool_df.dropna(subset=optimization_features).copy()
    if len(pool_df_clean) == 0:
        raise ValueError("Phase 2 無有效資料：優化特徵包含 NaN，無法計算平均值。")

    if len(pool_df_clean) < len(pool_df):
        removed = len(pool_df) - len(pool_df_clean)
        print(f"⚠️ Phase 2：剔除 {removed} 筆含 NaN 的優化特徵資料。")

    X_opt = pool_df_clean[optimization_features].values
    target_mean = np.mean(X_opt, axis=0)
    target_covariance = np.cov(X_opt, rowvar=False)

    print("靶心 (目標平均值)")
    for name, val in zip(optimization_features, target_mean):
        print(f" * {name}: {val:.2f}")

    print("共變異數矩陣 (3x3)")
    print(target_covariance)
    print("-" * 30)

    return target_mean, target_covariance

# phase 3 

def evaluate_pitch_confidence(current_pitch, target_mean, target_covariance, optimization_features=None):
    """
    Phase 3：計算馬哈拉諾比斯距離，並轉換為 0~100% 的落點信心度
    """
    if optimization_features is None:
        optimization_features = ['release_speed', 'release_spin_rate', 'spin_axis']

    # 1. 將數據轉成向量 (x)
    x = np.array([current_pitch[feat] for feat in optimization_features])
    
    # 2. 計算與完美靶心的「物理誤差」 (delta = x - mu)
    delta_vector = x - target_mean
    
    # 3. 計算共變異數矩陣的反矩陣 (用 pinv 避免數學報錯)
    cov_inv = np.linalg.pinv(target_covariance)
    
    # 4. 馬氏距離公式： D = sqrt( delta^T * cov_inv * delta )
    distance_squared = np.dot(np.dot(delta_vector, cov_inv), delta_vector)
    mahalanobis_dist = np.sqrt(max(0, distance_squared)) # max(0) 是保護機制，避免浮點數微小負值
    
    # 5. 轉換為 0~100 的信心度 (使用常態分佈衰減曲線)
    # -0.5 是一個常數，如果你覺得系統給分太嚴格，可以改成 -0.2；覺得太鬆可以改成 -1.0
    confidence_score = np.exp(-0.5 * (mahalanobis_dist ** 2)) * 100
    
    # 6. 把誤差打包成字典，準備交給 Phase 4 
    deltas_dict = {feat: diff for feat, diff in zip(optimization_features, delta_vector)}
    
    return confidence_score, mahalanobis_dist, deltas_dict


def run_hybrid_ai_system(raw_pitch_data: dict, target_zone: int, pitcher_profile: dict, df_database):
    """
    棒球分析系統 Pipeline
    輸入：一顆球的原始數據 -> 輸出：球種辨識結果 + 控球評分 + 教練建議
    """
    print("\n" + "=" * 50)
    print("⚾ [系統啟動] 接收到全新測速槍數據，開始解析...")
    print("=" * 50)
    
    # ----------------------------------------------------
    # 模型球種辨識
    # ----------------------------------------------------
    print("正在進行球種預測...")
    ml_result = predict_pitch(raw_pitch_data) 
    
    # 抽出預測的球種字串 (例如 'SL', 'FF', 'SI')
    detected_pitch_type = ml_result['predicted_pitch'] 
    
    print(f"這是一顆 【{detected_pitch_type}】 (信心 margin: {ml_result['margin']:.4f})")
    
    # ----------------------------------------------------
    # 球威評分 (Stuff+ Score)   
    # ----------------------------------------------------
    print("\n 評估球路軌跡品質 (Stuff+ Score)...")
    raw_pitch_data['pitch_type'] = detected_pitch_type
    
    try:
        stuff_score = evaluate_new_pitch(
            new_pitch=raw_pitch_data, 
            baseline_df=df_database, 
            config_matrix=PITCH_CONFIG
        )
        print(f"球威評分完成：綜合 PR 評分達 【 {stuff_score} 分 】")
    except Exception as e:
        print(f"球威計算時發生錯誤：{e}")
        stuff_score = None

    # ----------------------------------------------------
    # 馬氏靶心模型落點評分與物理診斷
    # ----------------------------------------------------
    print(f"\n (目標落點：{target_zone} 號位)...")
    
    try:
        # 1. 把辨識出的球種 (detected_pitch_type) 傳入你的 Phase 1
        gold_pool_df = build_success_similarity_pool(
            df=df_database, 
            target_pitch_type=detected_pitch_type, 
            target_zone=target_zone, 
            target_pitcher=pitcher_profile
        )
        
        # 2. 取出該球種的黃金標準 (Phase 2)
        opt_features = ['release_speed', 'release_spin_rate', 'spin_axis']
        target_mean, target_cov = run_phase2_analysis(gold_pool_df, opt_features)
        
        # 3. 計算這顆球的分數與誤差 (Phase 3)
        score, dist, deltas = evaluate_pitch_confidence(
            current_pitch=raw_pitch_data, 
            target_mean=target_mean, 
            target_covariance=target_cov,
            optimization_features=opt_features
        )
        
        print(f"這顆 {detected_pitch_type} 落入 {target_zone} 號位的預測信心度為 【{score:.1f}%】")
        
        
        return {
            "status": "success",
            "pitch_type": detected_pitch_type,
            "score": score,
            "errors": deltas
        }
        
    except ValueError as e:
        print(f"\n警告：{e}")
        return {"status": "error", "message": str(e)}


# ==========================================
# 測試區 
# ==========================================
my_pitcher = {
    'release_pos_x': -2.1,
    'release_pos_z': 5.5,
    'release_extension': 6.5
}
if __name__ == '__main__':
    try:
        # 新串接函式測試：
        raw_pitch_input = {
            'release_speed': 91,
            'release_spin_rate': 2010,
            'spin_axis': 225.0,
            'api_break_x_arm': -4.2,
            'api_break_z_with_gravity': 30.5,
            'pfx_x': -2.3,
            'pfx_z': 1.5,
            'ax': -6.8,
            'vx0': -4.9,
            'ay': 26.8,
            'vy0': -135.5,
            'arm_angle': 28.0,
            'p_throws': 'R'
        }

        result = run_hybrid_ai_system(
            raw_pitch_data=raw_pitch_input,
            target_zone=9,
            pitcher_profile=my_pitcher,
            df_database=df_clean
        )
    
        print("\n=== 新串接函式 run_hybrid_ai_system 測試結果 ===")
        print(result)
    except Exception as e:
        print("測試過濾器時發生錯誤：", e)
