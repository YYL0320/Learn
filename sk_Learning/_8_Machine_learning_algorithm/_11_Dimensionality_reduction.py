# # 导入必要的库
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.datasets import load_iris
# from sklearn.manifold import TSNE
#
# # -------------------------- 设置中文字体 start --------------------------
# plt.rcParams['font.sans-serif'] = [
#     # Windows 优先
#     'SimHei', 'Microsoft YaHei',
#     # macOS 优先
#     'PingFang SC', 'Heiti TC',
#     # Linux 优先
#     'WenQuanYi Micro Hei', 'DejaVu Sans'
# ]
# # 修复负号显示为方块的问题
# plt.rcParams['axes.unicode_minus'] = False
# # -------------------------- 设置中文字体 end --------------------------
#
# # 1. 加载经典的鸢尾花数据集（4个特征）
# iris = load_iris()
# X = iris.data  # 原始数据：150个样本，4个特征
# y = iris.target # 标签，用于可视化着色
#
# print(f"原始数据形状: {X.shape}")  # 输出: (150, 4)
#
# # # 2. 创建PCA模型，指定降维到2维
# # pca = PCA(n_components=2)
# #
# # # 3. 拟合模型（计算主成分）并转换数据
# # X_pca = pca.fit_transform(X)
#
# tsne = TSNE(n_components=2, perplexity=15, random_state=42)
# X_swiss_tsne = tsne.fit_transform(X)
#
# print(f"降维后数据形状: {X_swiss_tsne.shape}") # 输出: (150, 2)
# # print(f"各主成分解释的方差比例: {tsne.explained_variance_ratio_}")
# # 输出可能类似: [0.9246, 0.0530] 表示第一主成分保留了92.5%的信息，第二主成分保留了5.3%
#
# # 4. 可视化降维结果
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X_swiss_tsne[:, 0], X_swiss_tsne[:, 1], c=y, edgecolor='k', alpha=0.7)
# plt.xlabel('第一主成分 (PC1)')
# plt.ylabel('第二主成分 (PC2)')
# plt.title('PCA: 鸢尾花数据集降维可视化')
# plt.colorbar(scatter, label='鸢尾花种类')
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()

# #t-SNE
# # 导入必要的库
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.datasets import make_swiss_roll # 生成瑞士卷数据
#
# # -------------------------- 设置中文字体 start --------------------------
# plt.rcParams['font.sans-serif'] = [
#     # Windows 优先
#     'SimHei', 'Microsoft YaHei',
#     # macOS 优先
#     'PingFang SC', 'Heiti TC',
#     # Linux 优先
#     'WenQuanYi Micro Hei', 'DejaVu Sans'
# ]
# # 修复负号显示为方块的问题
# plt.rcParams['axes.unicode_minus'] = False
# # -------------------------- 设置中文字体 end --------------------------
#
# # 1. 生成一个非线性数据集：瑞士卷
# X_swiss, color = make_swiss_roll(n_samples=1000, noise=0.1)
# print(f"瑞士卷数据形状: {X_swiss.shape}") # (1000, 3)
#
# # 2. 使用PCA（线性方法）尝试降维
# pca = PCA(n_components=2)
# X_swiss_pca = pca.fit_transform(X_swiss)
#
# # 3. 使用t-SNE（非线性方法）降维
# # perplexity（困惑度）是t-SNE的关键参数，通常介于5到50之间，表示对局部/全局结构的平衡关注
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# X_swiss_tsne = tsne.fit_transform(X_swiss)
#
# # 4. 对比可视化
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))
#
# # PCA结果
# axes[0].scatter(X_swiss_pca[:, 0], X_swiss_pca[:, 1], c=color, cmap='viridis')
# axes[0].set_title('PCA降维结果')
# axes[0].set_xlabel('PC1')
# axes[0].set_ylabel('PC2')
#
# # t-SNE结果
# sc = axes[1].scatter(X_swiss_tsne[:, 0], X_swiss_tsne[:, 1], c=color, cmap='viridis')
# axes[1].set_title('t-SNE降维结果 (perplexity=30)')
# axes[1].set_xlabel('t-SNE 1')
# axes[1].set_ylabel('t-SNE 2')
#
# plt.colorbar(sc, ax=axes[1], label='瑞士卷的"高度"')
# plt.tight_layout()
# plt.show()

#手写数字降维
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import make_swiss_roll # 生成瑞士卷数据

# 1. 加载MNIST数据集（只取部分样本以加快速度）
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_mnist, y_mnist = mnist.data[:3000] / 255.0, mnist.target[:3000] # 归一化，取前3000个样本
print(f"MNIST数据形状: {X_mnist.shape}") # (3000, 784) -> 784维！

# 2. 先用PCA快速降到50维，去除大量噪声
pca = PCA(n_components=50)
X_mnist_pca = pca.fit_transform(X_mnist)
print(f"PCA后形状: {X_mnist_pca.shape}")

# 3. 再用t-SNE将50维数据降到2维进行可视化
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
X_mnist_tsne = tsne.fit_transform(X_mnist_pca)

# 4. 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_mnist_tsne[:, 0], X_mnist_tsne[:, 1],
                      c=y_mnist.astype(int), cmap='tab10', alpha=0.6, s=5)
plt.colorbar(scatter, ticks=range(10), label='手写数字')
plt.title('MNIST手写数字数据集经PCA预处理后的t-SNE可视化')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()
