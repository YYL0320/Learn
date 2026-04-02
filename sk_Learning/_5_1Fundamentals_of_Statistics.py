import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载数据集（这里使用 seaborn 自带的'tips'小费数据集）
df = sns.load_dataset('tips')
print("数据集前5行：")
print(df.head())
print(f"\n数据集形状：{df.shape}") # 查看行数和列数
print("\n基本信息：")
print(df.info())
print("\n描述性统计：")
print(df.describe())

# 2. 探索数值型变量：总账单（total_bill）和小费（tip）
print(f"\n总账单的均值：{df['total_bill'].mean():.2f}")
print(f"总账单的中位数：{df['total_bill'].median():.2f}")
print(f"总账单的标准差：{df['total_bill'].std():.2f}")
print(f"小费与总账单的相关系数：{df['tip'].corr(df['total_bill']):.3f}")

# 3. 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 3.1 总账单的直方图与密度估计
sns.histplot(df['total_bill'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Total Bill')
axes[0, 0].axvline(df['total_bill'].mean(), color='red', linestyle='--', label=f'Mean: {df["total_bill"].mean():.1f}')
axes[0, 0].axvline(df['total_bill'].median(), color='green', linestyle='--', label=f'Median: {df["total_bill"].median():.1f}')
axes[0, 0].legend()

# 3.2 小费与总账单的散点图（看相关性）
sns.scatterplot(data=df, x='total_bill', y='tip', hue='time', ax=axes[0, 1])
axes[0, 1].set_title('Tip vs Total Bill (Colored by Meal Time)')

# 3.3 按性别分组的小费箱线图（比较组间差异）
sns.boxplot(data=df, x='sex', y='tip', ax=axes[1, 0])
axes[1, 0].set_title('Tip Amount by Gender')

# 3.4 吸烟与否的账单均值柱状图
bill_by_smoker = df.groupby('smoker')['total_bill'].mean().reset_index()
sns.barplot(data=bill_by_smoker, x='smoker', y='total_bill', ax=axes[1, 1])
axes[1, 1].set_title('Average Total Bill by Smoking Status')
for index, row in bill_by_smoker.iterrows():
    axes[1, 1].text(index, row['total_bill']+0.5, f"{row['total_bill']:.1f}", ha='center')

plt.tight_layout()
plt.show()