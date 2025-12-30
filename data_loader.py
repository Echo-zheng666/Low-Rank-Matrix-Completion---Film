# data_loader.py
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(data_path):
    """
    加载MovieLens 1M数据集并构建用户-电影评分矩阵
    
    参数：
        data_path: str - ratings.dat文件的完整路径
    
    返回：
        rating_matrix: np.ndarray - 形状(n_users, n_movies)，用户-电影评分矩阵（缺失值为0）
        mask: np.ndarray - 形状(n_users, n_movies)，掩码矩阵（1=有评分，0=缺失）
        user_id_map: dict - 原始user_id到连续索引的映射
        movie_id_map: dict - 原始movie_id到连续索引的映射
    """
    # 读取评分数据（兼容Latin-1编码，跳过损坏行）
    try:
        ratings = pd.read_csv(
            data_path,
            sep='::',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python',
            encoding='latin-1',
            on_bad_lines='skip'
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 未找到文件：{data_path}，请检查路径是否正确")
    except Exception as e:
        raise Exception(f"❌ 读取数据失败：{str(e)}")
    
    # 映射原始ID到连续索引（避免ID不连续导致矩阵稀疏）
    user_ids = ratings['user_id'].unique()
    movie_ids = ratings['movie_id'].unique()
    user_id_map = {original_id: idx for idx, original_id in enumerate(user_ids)}
    movie_id_map = {original_id: idx for idx, original_id in enumerate(movie_ids)}
    
    # 初始化评分矩阵和掩码矩阵
    n_users = len(user_ids)
    n_movies = len(movie_ids)
    rating_matrix = np.zeros((n_users, n_movies), dtype=np.float32)  # 用float32节省内存
    mask = np.zeros_like(rating_matrix, dtype=np.float32)
    
    # 填充评分矩阵和掩码
    for _, row in ratings.iterrows():
        u_idx = user_id_map[row['user_id']]
        m_idx = movie_id_map[row['movie_id']]
        rating_matrix[u_idx, m_idx] = row['rating']
        mask[u_idx, m_idx] = 1.0
    
    # 输出加载信息
    total_ratings = np.sum(mask).astype(int)
    sparsity = 1 - (total_ratings / (n_users * n_movies))
    print(f"✅ 数据加载完成：")
    print(f"  - 用户数：{n_users} | 电影数：{n_movies}")
    print(f"  - 有效评分数：{total_ratings} | 矩阵稀疏度：{sparsity:.4f}")
    
    return rating_matrix, mask, user_id_map, movie_id_map


# 测试函数（可选，运行该文件时验证数据加载）
if __name__ == "__main__":
    # 替换为你的ratings.dat路径
    TEST_DATA_PATH = r"D:/大三上/最优化2/第二次大作业/ml-1m/ratings.dat"
    try:
        rating_matrix, mask, _, _ = load_and_preprocess_data(TEST_DATA_PATH)
        print(f"\n✅ 测试通过：评分矩阵形状={rating_matrix.shape}，掩码矩阵形状={mask.shape}")
    except Exception as e:
        print(f"\n❌ 测试失败：{str(e)}")