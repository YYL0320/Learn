from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


#L1
# 生成模拟数据
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 L1 正则化模型 (Lasso)， 设置正则化强度 alpha (即 λ)
lasso_model = Lasso(alpha=0.1) # alpha 越大， 惩罚越强， 更多权重为 0
lasso_model.fit(X_train, y_train)

# 查看模型系数（权重）， 观察稀疏性
print("Lasso 模型系数：")
for i, coef in enumerate(lasso_model.coef_):
    print(f"  特征 {i}: {coef:.4f}")

# 统计非零权重的数量
non_zero_count = sum(lasso_model.coef_ != 0)
print(f"\n非零权重的特征数量： {non_zero_count} / {X.shape[1]}")


from sklearn.linear_model import Ridge

# 创建 L2 正则化模型 (Ridge)
ridge_model = Ridge(alpha=1.0) # alpha 即 λ
ridge_model.fit(X_train, y_train)

# 查看模型系数， 观察权重衰减
print("Ridge 模型系数：")
for i, coef in enumerate(ridge_model.coef_):
    print(f"  特征 {i}: {coef:.4f}")

# 对比 Lasso 和 Ridge 的系数差异
print("\n系数对比 (Lasso vs Ridge):")
print("特征 | Lasso 系数 | Ridge 系数")
print("-" * 35)
for i in range(len(lasso_model.coef_)):
    print(f"{i:4d} | {lasso_model.coef_[i]:11.4f} | {ridge_model.coef_[i]:11.4f}")

#Dropout (用于神经网络)

#早停法