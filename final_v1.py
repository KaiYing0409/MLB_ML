import pandas as pd
import numpy as np


#%% 1. 指定檔案路徑 
file_path = "statcast_bat_tracking_2024_2025.csv" 
print(f"正在讀取資料: {file_path} ...")

df = pd.read_csv(file_path)

# 2. 定義核心欄位並篩選 (減少記憶體佔用)
columns_to_keep = [
    'pitch_type',         # 球種
    'release_pos_x',      # 出手點橫向位置
    'release_pos_z',      # 出手高度
    'release_extension',  # 延伸距離
    'zone',               # 實際落點區域 (1~14)
    'release_speed',      # 球速
    'release_spin_rate',  # 轉速
    'spin_axis'           # 轉軸
]
df_clean = df[columns_to_keep].copy()

# 3. 剔除關鍵欄位有缺失值 (NaN) 的無效數據
initial_count = len(df_clean)
df_clean = df_clean.dropna(subset=['pitch_type', 'release_pos_x', 'release_pos_z', 'release_extension', 'zone'])
final_count = len(df_clean)

# 4. 檢視結果
print(f"✅ 資料處理完成！")
print(f"原始資料筆數: {initial_count}")
print(f"有效資料筆數: {final_count}")
print(f"剔除了 {initial_count - final_count} 筆破圖數據。")
print("-" * 30)

#%%
# 預覽前五筆資料
print(df_clean.head())


#phase 1 : 建立「成功落點」+「身體條件相似」的黃金基準池
def build_success_similarity_pool(df, target_pitch_type, target_zone, target_pitcher, top_k=100):
    """
    先鎖定「成功落點」，再找「身體條件最相似」的投手，建立基準池。
    
    參數:
    df: 清理過後的 Statcast DataFrame (例如 df_clean)
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
        print(f"⚠️ 警告：符合條件的總球數 ({len(df_success)}) 少於設定的 top_k ({top_k})。將取用全部符合的數據。")
        top_k = len(df_success)
    else:
        print(f"✅ 第一層過濾完成：共有 {len(df_success)} 顆成功的球。")

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

    X_opt = pool_df[optimization_features].values
    target_mean = np.mean(X_opt, axis=0)
    target_covariance = np.cov(X_opt, rowvar=False)

    print("靶心 (目標平均值)")
    for name, val in zip(optimization_features, target_mean):
        print(f" * {name}: {val:.2f}")

    print("共變異數矩陣 (3x3)")
    print(target_covariance)
    print("-" * 30)

    return target_mean, target_covariance


# ==========================================
# 測試區 
# ==========================================
my_pitcher = {
    'release_pos_x': -2.1,
    'release_pos_z': 5.8,
    'release_extension': 6.2
}

# 執行過濾器
try:
    pool_df = build_success_similarity_pool(
        df=df_clean,
        target_pitch_type='SL',
        target_zone=9,
        target_pitcher=my_pitcher,
        top_k=100
    )

    print("\n=== 測試過濾器結果 (前 100 筆) ===")
    print(f"目標投手出手特徵：{my_pitcher}")
    print(f"找到的相似成功樣本：{len(pool_df)}")
    print(pool_df[['pitch_type', 'zone', 'release_pos_x', 'release_pos_z', 'release_extension', 'physical_distance']].head(10))

    # phase 2 :
    run_phase2_analysis(pool_df)
except Exception as e:
    print("測試過濾器時發生錯誤：", e)
