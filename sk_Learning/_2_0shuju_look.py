import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# 或者从本地文件加载
df = pd.read_csv('iris.csv')

# 2. 查看数据的前几行 - 第一印象
print("数据的前5行：")
print(df.head())
print("\n" + "="*50 + "\n")

# 3. 查看数据的整体信息：行数、列数、数据类型、内存占用
print("数据集的基本信息：")
print(df.info())
print("\n" + "="*50 + "\n")

# 4. 查看数据的形状（多少行，多少列）
print(f"数据集形状：{df.shape}") # 输出 (行数， 列数)
print(f"共有 {df.shape[0]} 条样本， {df.shape[1]} 个特征。")


# 1. 检查缺失值
print("各特征缺失值数量：")
print(df.isnull().sum())
print("\n" + "="*50 + "\n")

# 如果缺失值很多，可以计算缺失比例
missing_ratio = df.isnull().sum() / len(df) * 100
print("各特征缺失值比例（%）:")
print(missing_ratio)
print("\n" + "="*50 + "\n")

# 2. 检查数值型特征的统计摘要 - 可以发现异常值的线索
print("数值型特征的统计描述：")
print(df.describe())


# 设置图表风格
sns.set(style="whitegrid")

# 1. 单变量分布 - 了解每个特征自身的分布情况
fig, axes = plt.subplots(2, 2, figsize=(12, 8)) # 创建2x2的画布
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
colors = ['skyblue', 'lightgreen', 'salmon','gold']

for i, (ax, feature, color) in enumerate(zip(axes.flat, features, colors)):
    # 绘制直方图（分布）与核密度估计曲线
    sns.histplot(df[feature], kde=True, ax=ax, color=color, bins=20)
    ax.set_title(f'{feature} 的分布', fontsize=14)
    ax.set_xlabel(feature)
    ax.set_ylabel('频数')

plt.tight_layout()
plt.show()

# 2. 箱线图 - 查看数据分布与异常值（更直观）
plt.figure(figsize=(10, 6))
# 选择数值列绘制箱线图
df_box = df.drop(columns=['species']) # 假设'species'是文本标签列，先去掉
sns.boxplot(data=df_box)
plt.title('各数值特征的箱线图（查看分布与异常值）', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# 3. 变量间关系 - 散点图矩阵
print("\n绘制特征间关系的散点图矩阵...（这能帮助我们发现特征之间的关联）")
# 使用Seaborn的pairplot， hue参数可以根据类别着色（如鸢尾花的品种）
sns.pairplot(df, hue='species', height=2.5)
plt.suptitle('特征关系散点图矩阵（按种类着色）', y=1.02, fontsize=16)
plt.show()

# 4. 相关性热力图 - 量化特征间的线性关系
plt.figure(figsize=(8, 6))
# 计算数值特征之间的相关系数
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('特征相关性热力图', fontsize=14)
plt.show()
