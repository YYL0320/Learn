# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# import pandas as pd
#
# # 假设我们有一个简单的房价数据集
# data = {
#     '面积': [50, 60, 80, 100, 120],
#     '房价': [150, 180, 240, 300, 350]
# }
# df = pd.DataFrame(data)
#
# # 特征和标签
# X = df[['面积']]
# y = df['房价']
#
# # 数据分割
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 训练线性回归模型
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # 预测
# y_pred = model.predict(X_test)
#
# print(f"预测的房价: {y_pred}")


#多项式回归
# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 设置随机种子，确保每次运行结果一致
np.random.seed(42)

# 创建模拟数据：y 是 x 的二次函数加上一些随机噪声
X = 6 * np.random.rand(100, 1) - 3  # 生成100个在[-3, 3)区间的随机数
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)  # y = 0.5x² + x + 2 + 噪声

# 可视化原始数据
plt.scatter(X, y, s=10, alpha=0.7, label='原始数据')
plt.xlabel('X')
plt.ylabel('y')
plt.title('模拟的非线性数据')
plt.legend()
plt.show()

# 1. 创建多项式特征
# 参数 degree 决定了多项式的阶数，这里我们尝试2阶
poly_features = PolynomialFeatures(degree=2, include_bias=False)
# 将原始特征X转换为包含X和X^2的新特征矩阵X_poly
X_poly = poly_features.fit_transform(X)

print(f"原始X的形状: {X.shape}")
print(f"转换后X_poly的形状: {X_poly.shape}")
print(f"前5行X_poly数据:\n{X_poly[:5]}")
# 输出显示，X_poly 有两列：第一列是X，第二列是X^2

# 2. 在转换后的特征上训练线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)  # 使用X_poly，而不是原始的X

# 3. 查看学到的模型参数（权重和偏置）
print(f"\n模型参数（权重w1, w2）: {lin_reg.coef_.ravel()}")
print(f"模型偏置（截距b）: {lin_reg.intercept_}")
# 输出结果应接近我们生成数据时用的参数 [1, 0.5] 和 2

# 为了画出平滑的曲线，需要生成一组均匀分布的点
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
# 对这组新点同样进行多项式特征转换
X_new_poly = poly_features.transform(X_new)
# 用模型进行预测
y_new = lin_reg.predict(X_new_poly)

# 开始绘图
plt.scatter(X, y, s=10, alpha=0.7, label='训练数据')
plt.plot(X_new, y_new, 'r-', linewidth=2, label='多项式回归拟合 (degree=2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('二次多项式回归拟合效果')
plt.legend()
plt.show()

#交叉验证
from sklearn.model_selection import cross_val_score

# 测试一系列阶数
degrees_to_try = range(1, 11)
cv_scores = []

for degree in degrees_to_try:
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    lin_reg = LinearRegression()
    # 使用5折交叉验证，以负均方误差作为评分（sklearn约定：分数越高越好，所以用负MSE）
    scores = cross_val_score(lin_reg, X_poly, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())  # 取平均并转回正数MSE

# 找到使交叉验证误差最小的阶数
best_degree = degrees_to_try[np.argmin(cv_scores)]
print(f"根据交叉验证，最佳阶数是: {best_degree}")

# 可视化交叉验证误差随阶数的变化
plt.plot(degrees_to_try, cv_scores, 'bo-')
plt.xlabel('多项式阶数')
plt.ylabel('5折交叉验证平均MSE')
plt.title('交叉验证选择最佳阶数')
plt.axvline(x=best_degree, color='r', linestyle='--', label=f'最佳阶数={best_degree}')
plt.legend()
plt.grid(True)
plt.show()