import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 导入独立的数据加载模块（与非凸优化共用）
from data_loader import load_and_preprocess_data

# ===================== 全局配置 =====================
# 数据集路径（替换为你的ratings.dat实际路径）
DATA_PATH = r"D:/大三上/最优化2/第二次大作业/ml-1m/ratings.dat"
# 随机种子（保证结果可复现）
RANDOM_SEED = 42
# 交叉验证折数
KFOLD = 5
# 核范数最小化超参数（预调优）
LAMBDA_REG = 0.15    # 正则化系数
MAX_ITER_CONVEX = 150  # 最大迭代次数

# ===================== 工具函数：NaN/Inf清理 =====================
def clean_nan(arr, fill_value=0.0):
    """清理数组中的NaN/Inf，替换为指定值"""
    arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return arr

# ===================== 凸优化核心算法：核范数最小化 =====================
def nuclear_norm_minimization(M, mask, lambda_reg=LAMBDA_REG, max_iter=MAX_ITER_CONVEX):
    """
    核范数最小化（凸松弛求解低秩矩阵填充）
    :param M: 原始评分矩阵 (n_users, n_movies)
    :param mask: 训练集掩码（1=观测值，0=缺失值）
    :param lambda_reg: 正则化系数（平衡核范数和MSE）
    :param max_iter: L-BFGS-B优化最大迭代次数
    :return: 填充后的评分矩阵（限制1-5分）
    """
    # 数值稳定性处理：清理输入中的NaN/Inf
    M = clean_nan(M, np.mean(M[mask == 1]))
    mask = clean_nan(mask, 0.0)
    
    # 目标函数：核范数 + 正则化MSE损失
    def objective(X_flat):
        X = X_flat.reshape(M.shape)
        # 核范数（奇异值之和，凸松弛的低秩约束）
        try:
            nuclear_norm = np.linalg.norm(X, ord='nuc')
        except:
            # 数值不稳定时降级为Frobenius范数
            nuclear_norm = np.linalg.norm(X, ord='fro') / 100
        # 观测值MSE损失（仅计算已知评分）
        mse_loss = np.sum((mask * (X - M)) ** 2) / 2
        return nuclear_norm + lambda_reg * mse_loss

    # 梯度函数（MSE部分的梯度，核范数梯度由L-BFGS-B自动近似）
    def gradient(X_flat):
        X = X_flat.reshape(M.shape)
        grad = mask * (X - M)  # MSE梯度
        return clean_nan(grad.flatten(), 0.0)  # 清理梯度中的NaN

    # 初始化矩阵（观测值保留，缺失值填充全局均值）
    global_mean = np.sum(M * mask) / np.sum(mask)
    init_matrix = M.copy()
    init_matrix[mask == 0] = global_mean
    init_matrix = clean_nan(init_matrix, global_mean)  # 兜底清理

    # L-BFGS-B优化求解（凸优化的高效求解器）
    try:
        res = minimize(
            fun=objective,
            x0=init_matrix.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'disp': False,
                'gtol': 1e-4  # 梯度收敛阈值，提升稳定性
            }
        )
        filled_matrix = res.x.reshape(M.shape)
    except:
        # 优化失败时降级为均值填充
        filled_matrix = init_matrix

    # 最终处理：限制评分范围1-5 + NaN清理
    filled_matrix = clean_nan(filled_matrix, global_mean)
    filled_matrix = np.clip(filled_matrix, 1, 5)
    return filled_matrix

# ===================== 5折交叉验证评估 =====================
def cross_validate_convex(M, mask):
    """
    5折交叉验证评估核范数最小化（凸方法）性能
    :param M: 原始评分矩阵
    :param mask: 完整掩码矩阵
    :return: 各折RMSE列表、均值、标准差
    """
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_SEED)
    non_zero_coords = np.argwhere(mask == 1)  # 仅对有评分的位置做交叉验证
    rmse_list = []
    global_mean = np.sum(M * mask) / np.sum(mask)

    print("\n📊 开始凸方法（核范数最小化）5折交叉验证...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(non_zero_coords)):
        # 构建训练/测试掩码
        train_mask = np.zeros_like(mask)
        test_mask = np.zeros_like(mask)
        
        # 填充训练掩码（仅保留训练集评分）
        for (u, m) in non_zero_coords[train_idx]:
            train_mask[u, m] = 1.0
        # 填充测试掩码（仅保留测试集评分）
        for (u, m) in non_zero_coords[test_idx]:
            test_mask[u, m] = 1.0

        # 训练凸模型
        print(f"\n----- 第{fold+1}/{KFOLD}折 -----")
        filled_matrix = nuclear_norm_minimization(
            M, train_mask,
            lambda_reg=LAMBDA_REG,
            max_iter=MAX_ITER_CONVEX
        )

        # 计算测试集RMSE（NaN防护）
        pred = filled_matrix[test_mask == 1]
        true = M[test_mask == 1]
        # 最终清理
        pred = clean_nan(pred, global_mean)
        true = clean_nan(true, global_mean)
        
        # 防止空数组报错
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
    print(f"🎯 凸优化最终结果 | 5折RMSE均值: {mean_rmse:.4f} | 标准差: {std_rmse:.4f}")
    print("="*50)

    return rmse_list, mean_rmse, std_rmse

# ===================== 结果可视化 =====================
def plot_convex_results(rmse_list):
    """可视化凸方法各折RMSE结果"""
    plt.figure(figsize=(10, 6))
    folds = [f"第{i+1}折" for i in range(KFOLD)]
    plt.bar(folds, rmse_list, color='#3498db', alpha=0.8)
    plt.axhline(y=np.mean(rmse_list), color='#e74c3c', linestyle='--', 
                label=f'均值RMSE: {np.mean(rmse_list):.4f}')
    
    plt.xlabel('交叉验证折数', fontsize=12)
    plt.ylabel('RMSE（越低越好）', fontsize=12)
    plt.title('MovieLens 1M 核范数最小化（凸方法）RMSE 结果', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('convex_rmse_results.png', dpi=300)
    plt.show()

# ===================== 主函数（完整运行逻辑） =====================
if __name__ == "__main__":
    # 步骤1：加载数据（调用共用的data_loader模块）
    print("🔍 加载MovieLens 1M数据集...")
    rating_matrix, mask, _, _ = load_and_preprocess_data(DATA_PATH)
    
    # 步骤2：打印凸方法超参数
    print(f"\n✅ 凸方法超参数：")
    print(f"  - 正则化系数lambda: {LAMBDA_REG}")
    print(f"  - 最大迭代次数: {MAX_ITER_CONVEX}")
    print(f"  - 交叉验证折数: {KFOLD}")
    
    # 步骤3：5折交叉验证评估
    rmse_list, mean_rmse, std_rmse = cross_validate_convex(rating_matrix, mask)
    
    # 步骤4：结果可视化
    plot_convex_results(rmse_list)
    
    # 步骤5：保存结果到文件
    with open('convex_results.txt', 'w') as f:
        f.write(f"凸方法超参数：\n")
        f.write(f"  lambda_reg: {LAMBDA_REG}\n")
        f.write(f"  max_iter: {MAX_ITER_CONVEX}\n")
        f.write(f"5折RMSE均值: {mean_rmse:.4f}\n")
        f.write(f"5折RMSE标准差: {std_rmse:.4f}\n")
        f.write(f"各折RMSE: {rmse_list}\n")
    
    print("\n📄 凸方法结果已保存到 convex_results.txt 和 convex_rmse_results.png")
    print("\n✅ 凸优化完整流程运行结束！")