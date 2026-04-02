# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# -------------------------- 设置中文字体 start --------------------------
plt.rcParams['font.sans-serif'] = [
    # Windows 优先
    'SimHei', 'Microsoft YaHei',
    # macOS 优先
    'PingFang SC', 'Heiti TC',
    # Linux 优先
    'WenQuanYi Micro Hei', 'DejaVu Sans'
]
# 修复负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# -------------------------- 设置中文字体 end --------------------------

# 1. 创建一个人工数据集
# 我们生成 300 个样本点，它们天然地围绕 4 个中心分布（方便我们观察）
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
# X 是特征数据，y_true 是真实的类别标签（仅用于最后对比，聚类算法不会用到它）

# # 2. 可视化原始数据
# plt.scatter(X[:, 0], X[:, 1], s=50) # s 是点的大小
# plt.title("原始未标记数据")
# plt.show()

# # 3. 应用 K-Means 聚类
# # 指定要聚成 4 类
# kmeans = KMeans(n_clusters=4, random_state=0, n_init='auto')
# # 拟合模型并预测每个样本的簇标签
# y_kmeans = kmeans.fit_predict(X)
#
# # 4. 获取质心坐标
# centroids = kmeans.cluster_centers_
#
# # 5. 可视化聚类结果
# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
# # 用不同颜色标注不同簇的样本点
#
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.8, marker='X')
# # 用红色大叉标出质心位置，alpha 是透明度
# plt.title("K-Means 聚类结果 (K=4)")
# plt.show()
#
# # 打印前10个样本的预测簇标签
# print("前10个样本的簇标签:", y_kmeans[:10])
# # 打印质心坐标
# print("四个簇的质心坐标:\n", centroids)

# 肘部法则示例：计算不同K值下的 inertia
inertias = []
K_range = range(1, 11) # 测试 K 从 1 到 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    kmeans.fit(X)
    inertias.append(kmeans.inertia_) # inertia_ 属性即 SSE

# 绘制肘部曲线
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('簇的数量 K')
plt.ylabel('Inertia (SSE)')
plt.title('肘部法则寻找最佳 K 值')
plt.axvline(x=4, color='r', linestyle='--', alpha=0.5) # 标记我们已知的 K=4
plt.show()