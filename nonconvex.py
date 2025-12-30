import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
# 导入数据加载模块
from data_loader import load_and_preprocess_data

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 全局配置 =====================
# 数据集路径（请替换为你的ratings.dat实际路径）
DATA_PATH = r"D:/大三上/最优化2/第二次大作业/ml-1m/ratings.dat"
# 随机种子（保证结果可复现）
RANDOM_SEED = 42
# 交叉验证折数
KFOLD = 5

# ===================== 工具函数：NaN清理 + 数值稳定 =====================
def clean_nan(arr, fill_value=0.0):
    """清理数组中的NaN/Inf，替换为指定值"""
    arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return arr

def clip_gradient(grad, max_norm=1.0):
    """梯度裁剪，防止数值爆炸"""
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        grad = grad * max_norm / norm
    return grad

# ===================== 2. 改进的非凸优化核心算法 =====================
def improved_spectral_init(M, mask, rank):
    """改进的谱初始化：均值填充+正则化SVD（NaN防护+数值稳定）"""
    # 基础配置
    global_mean = np.mean(M[mask == 1])
    n_users, n_movies = M.shape
    
    # 1. 计算用户/电影均值（全量NaN防护）
    user_sum = np.sum(M * mask, axis=1)
    user_count = np.sum(mask, axis=1)
    user_mean = np.where(user_count > 0, user_sum / user_count, global_mean)
    
    movie_sum = np.sum(M * mask, axis=0)
    movie_count = np.sum(mask, axis=0)
    movie_mean = np.where(movie_count > 0, movie_sum / movie_count, global_mean)
    
    # 2. 清理均值中的NaN/Inf
    user_mean = clean_nan(user_mean, global_mean)
    movie_mean = clean_nan(movie_mean, global_mean)
    
    # 3. 广播填充缺失值（维度匹配+NaN防护）
    filled_init = M.copy()
    # 用户均值广播
    user_mean_broadcast = np.tile(user_mean.reshape(-1, 1), (1, n_movies))
    filled_init = np.where(mask == 1, filled_init, user_mean_broadcast)
    # 电影均值混合填充
    movie_mean_broadcast = np.tile(movie_mean.reshape(1, -1), (n_users, 1))
    filled_init = np.where(
        mask == 1, 
        filled_init, 
        0.5 * filled_init + 0.5 * movie_mean_broadcast
    )
    
    # 4. 最终清理（防止填充过程中产生NaN）
    filled_init = clean_nan(filled_init, global_mean)
    
    # 5. 正则化SVD（数值稳定）
    try:
        U, S, Vt = np.linalg.svd(filled_init, full_matrices=False)
        # 奇异值裁剪+清理
        S = clean_nan(S, 0.0)
        S = np.clip(S, 0, np.percentile(S, 95))  # 收缩异常奇异值
        # 处理秩超过奇异值数量的情况
        if rank > len(S):
            rank = len(S)
        # 构建初始低秩矩阵
        X_init = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    except:
        # SVD失败时降级初始化（全局均值填充）
        X_init = np.full_like(M, global_mean)
    
    # 最终裁剪+NaN清理
    X_init = clean_nan(X_init, global_mean)
    X_init = np.clip(X_init, 1, 5)
    return X_init

def two_stage_nonconvex_optimized(M, mask, rank=80, lr=0.02, max_iter=1000, reg=0.001):
    """优化后的两阶段非凸低秩矩阵填充（全链路NaN防护）"""
    n_users, n_movies = M.shape
    global_mean = np.mean(M[mask == 1])
    
    # 阶段1：改进谱初始化（NaN防护）
    X_init = improved_spectral_init(M, mask, rank)
    X_init = clean_nan(X_init, global_mean)
    
    # 初始化U/V（数值稳定）
    try:
        U_init, S_init, Vt_init = np.linalg.svd(X_init, full_matrices=False)
        S_init = clean_nan(S_init, 0.0)
        if rank > len(S_init):
            rank = len(S_init)
        U = U_init[:, :rank] @ np.diag(np.sqrt(np.clip(S_init[:rank], 1e-6, None)))
        V = Vt_init[:rank, :].T @ np.diag(np.sqrt(np.clip(S_init[:rank], 1e-6, None)))
    except:
        # SVD失败时随机初始化（带正则）
        U = np.random.normal(0, 0.1, (n_users, rank)) * reg
        V = np.random.normal(0, 0.1, (n_movies, rank)) * reg
    
    # 清理U/V中的NaN
    U = clean_nan(U, 0.0)
    V = clean_nan(V, 0.0)
    
    # 阶段2：带防护的梯度下降
    for iter in range(max_iter):
        # 预测矩阵（NaN防护）
        X_pred = U @ V.T
        X_pred = clean_nan(X_pred, global_mean)
        
        # 残差计算（仅观测值）
        residual = mask * (X_pred - M)
        residual = clean_nan(residual, 0.0)
        
        # 梯度计算+正则化+裁剪
        grad_U = residual @ V + reg * U
        grad_V = residual.T @ U + reg * V
        # 梯度裁剪防止数值爆炸
        grad_U = clip_gradient(grad_U, max_norm=10.0)
        grad_V = clip_gradient(grad_V, max_norm=10.0)
        # 清理梯度中的NaN
        grad_U = clean_nan(grad_U, 0.0)
        grad_V = clean_nan(grad_V, 0.0)
        
        # 学习率衰减+更新
        lr_decay = lr / (1 + 0.001 * iter)
        U -= lr_decay * grad_U
        V -= lr_decay * grad_V
        
        # 清理U/V
        U = clean_nan(U, 0.0)
        V = clean_nan(V, 0.0)
    
    # 最终填充矩阵（全量防护）
    X_filled = U @ V.T
    X_filled = clean_nan(X_filled, global_mean)
    X_filled = np.clip(X_filled, 1, 5)
    return X_filled

# ===================== 4. 5折交叉验证评估 =====================
def cross_validate_nonconvex(M, mask, best_params):
    """5折交叉验证评估非凸优化模型（NaN防护）"""
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_SEED)
    non_zero_coords = np.argwhere(mask == 1)
    rmse_list = []
    global_mean = np.mean(M[mask == 1])
    
    print("\n📊 开始5折交叉验证...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(non_zero_coords)):
        # 构建训练/测试掩码
        train_mask = np.zeros_like(mask)
        test_mask = np.zeros_like(mask)
        
        for (u, m) in non_zero_coords[train_idx]:
            train_mask[u, m] = 1.0
        for (u, m) in non_zero_coords[test_idx]:
            test_mask[u, m] = 1.0
        
        # 训练非凸模型
        print(f"\n----- 第{fold+1}/{KFOLD}折 -----")
        X_filled = two_stage_nonconvex_optimized(
            M, train_mask,
            rank=best_params['rank'],
            lr=best_params['lr'],
            max_iter=best_params['max_iter'],
            reg=best_params['reg']
        )
        
        # 计算测试集RMSE（NaN防护）
        pred = X_filled[test_mask == 1]
        true = M[test_mask == 1]
        # 最终清理预测值和真实值中的NaN
        pred = clean_nan(pred, global_mean)
        true = clean_nan(true, global_mean)
        
        # 计算RMSE（防止空数组）
        if len(pred) == 0 or len(true) == 0:
            rmse = 0.0
        else:
            rmse = np.sqrt(mean_squared_error(true, pred))
        
        rmse_list.append(rmse)
        print(f"第{fold+1}折RMSE: {rmse:.4f}")
    
    # 统计结果
    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)
    print("\n" + "="*50)
    print(f"🎯 非凸优化最终结果 | 5折RMSE均值: {mean_rmse:.4f} | 标准差: {std_rmse:.4f}")
    print("="*50)
    
    return rmse_list, mean_rmse, std_rmse

# ===================== 5. 结果可视化 =====================
def plot_results(rmse_list):
    """可视化各折RMSE结果"""
    plt.figure(figsize=(10, 6))
    folds = [f"第{i+1}折" for i in range(KFOLD)]
    plt.bar(folds, rmse_list, color='#e74c3c', alpha=0.8)
    plt.axhline(y=np.mean(rmse_list), color='#3498db', linestyle='--', label=f'均值RMSE: {np.mean(rmse_list):.4f}')
    
    plt.xlabel('交叉验证折数', fontsize=12)
    plt.ylabel('RMSE（越低越好）', fontsize=12)
    plt.title('MovieLens 1M 非凸优化矩阵填充 RMSE 结果', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('nonconvex_rmse_results.png', dpi=300)
    plt.show()

# ===================== 主函数（完整运行逻辑） =====================
if __name__ == "__main__":
    # 步骤1：加载数据
    rating_matrix, mask, _, _ = load_and_preprocess_data(DATA_PATH)
    
    # 步骤2：手动设置预调优参数（降低学习率，提升稳定性）
    best_params = {
        'rank': 60,          # 降低秩提升数值稳定性
        'lr': 0.01,          # 降低学习率避免梯度爆炸
        'reg': 0.001,        # 正则化防止过拟合
        'max_iter': 800      # 减少迭代次数，加快运行
    }
    print(f"\n✅ 使用预调优参数：{best_params}")
    
    # 步骤3：5折交叉验证评估
    rmse_list, mean_rmse, std_rmse = cross_validate_nonconvex(rating_matrix, mask, best_params)
    
    # 步骤4：结果可视化
    plot_results(rmse_list)
    
    # 步骤5：保存最优参数和结果
    with open('nonconvex_results.txt', 'w') as f:
        f.write(f"最优超参数: {best_params}\n")
        f.write(f"5折RMSE均值: {mean_rmse:.4f}\n")
        f.write(f"5折RMSE标准差: {std_rmse:.4f}\n")
        f.write(f"各折RMSE: {rmse_list}\n")
    print("\n📄 结果已保存到 nonconvex_results.txt 和 nonconvex_rmse_results.png")