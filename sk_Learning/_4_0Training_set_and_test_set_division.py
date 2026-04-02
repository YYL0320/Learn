# # 示例：在分类问题中使用分层抽样
# from sklearn.model_selection import train_test_split
#
# # 假设 y 是分类标签
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)# stratify=按照 y 的类别分布来进行分层抽样
#
# # 检查划分后的类别比例
# from collections import Counter
# print("原始数据类别分布：", Counter(y))
# print("训练集类别分布：", Counter(y_train))
# print("测试集类别分布：", Counter(y_test))

# # 示例：时间序列数据的顺序划分,用前百分之八十为训练集
# split_index = int(len(X) * 0.8) # 计算80%位置的索引
#
# X_train, X_test = X[:split_index], X[split_index:]
# y_train, y_test = y[:split_index], y[split_index:]
#
# print(f"训练集时间范围：前 {split_index} 个样本")
# print(f"测试集时间范围：后 {len(X) - split_index} 个样本")

#k则交叉验证

# # 示例：使用5折交叉验证评估模型
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression
#
# model = LogisticRegression()
# scores = cross_val_score(model, X, y, cv=5) # cv=5 表示5折交叉验证
#
# print(f"各折得分：{scores}")
# print(f"平均得分：{scores.mean():.4f} (+/- {scores.std()*2:.4f})") # 输出平均分和标准差

# 1. 导入必要的库
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 2. 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target
print(f"数据集形状：特征 {X.shape}, 标签 {y.shape}")

# 3. 简单随机划分 (80%训练， 20%测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
print(f"随机划分 -> 训练集：{X_train.shape}， 测试集：{X_test.shape}")

# 4. 分层随机划分
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
print(f"分层划分 -> 训练集：{X_train_s.shape}， 测试集：{X_test_s.shape}")

# 5. 检查分层效果
print("\n原始数据类别分布：", np.bincount(y))
print("随机划分后测试集分布：", np.bincount(y_test)) # 可能不均衡
print("分层划分后测试集分布：", np.bincount(y_test_s)) # 应与原始分布成比例