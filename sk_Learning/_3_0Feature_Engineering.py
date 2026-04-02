# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import numpy as np
#
# # 示例数据
# data = np.array([[1000, 25],
#                  [1500, 30],
#                  [800, 20],
#                  [1200, 28]])
#
# # 标准化
# scaler_standard = StandardScaler()
# data_standardized = scaler_standard.fit_transform(data)
# print("标准化后的数据（均值~0， 标准差~1）：")
# print(data_standardized)
# print(f"均值： {data_standardized.mean(axis=0)}")
# print(f"标准差： {data_standardized.std(axis=0)}")
#
# # 归一化
# scaler_minmax = MinMaxScaler()
# data_normalized = scaler_minmax.fit_transform(data)
# print("\n归一化后的数据（范围[0,1]）：")
# print(data_normalized)

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#
# # 示例数据
# df_cat = pd.DataFrame({'城市': ['北京', '上海', '广州', '北京', '深圳']})
#
# # 标签编码
# le = LabelEncoder()
# df_cat['城市_标签编码'] = le.fit_transform(df_cat['城市'])
# print("标签编码结果：")
# print(df_cat)
#
# # 独热编码
# # 方法1: 使用 pandas 的 get_dummies
# df_onehot_pd = pd.get_dummies(df_cat['城市'], prefix='城市')
# print("\n使用 pandas 进行独热编码：")
# print(df_onehot_pd)
#
# # 方法2: 使用 sklearn 的 OneHotEncoder (更常用于管道)
# ohe = OneHotEncoder(sparse_output=False) # sparse_output=False 返回数组而非稀疏矩阵
# encoded_array = ohe.fit_transform(df_cat[['城市']]) # 注意输入是二维的
# print("\n使用 sklearn 进行独热编码（数组形式）：")
# print(encoded_array)
# print("新特征名称：", ohe.get_feature_names_out())


from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据集
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 训练一个随机森林模型，它会计算特征重要性
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 获取特征重要性
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    '特征': data.feature_names,
    '重要性': importances
}).sort_values('重要性', ascending=False)

print("特征重要性排序：")
print(feature_importance_df.head(10)) # 查看最重要的10个特征

# 可视化
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['特征'][:10], feature_importance_df['重要性'][:10])
plt.xlabel('特征重要性')
plt.title('Top 10 特征重要性')
plt.gca().invert_yaxis() # 让最重要的在顶部
plt.show()

# 假设我们选择重要性大于0.03的特征
selected_features = feature_importance_df[feature_importance_df['重要性'] > 0.03]['特征'].tolist()
print(f"\n筛选出的特征： {selected_features}")