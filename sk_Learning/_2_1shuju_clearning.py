# 导入必要的库
import pandas as pd
import numpy as np

# 加载数据集
df = pd.read_csv('iris.csv') # 请替换为你的文件路径

# 查看数据的基本信息和前几行
print("数据集形状（行，列）:", df.shape)
print("\n数据前5行：")
print(df.head())
print("\n数据基本信息：")
print(df.info())
print("\n数据统计描述：")
print(df.describe())

# 1. 检查缺失值
print("各列缺失值数量：")
print(df.isnull().sum())

# 2. 处理缺失值 - 方法一：删除
# 删除任何包含缺失值的行（适用于缺失值很少的情况）
df_dropped = df.dropna()
print(f"\n删除缺失值后，数据形状: {df_dropped.shape}")

# 3. 处理缺失值 - 方法二：填充
# 更常用的方法是根据列的特性进行填充
df_filled = df.copy()

# 对于数值型列（如'年龄'），用中位数填充（比均值更抗异常值影响）
if '年龄' in df_filled.columns:
    df_filled['年龄'].fillna(df_filled['年龄'].median(), inplace=True)

# 对于分类列（如'城市'），用众数（最频繁出现的值）填充
if '城市' in df_filled.columns:
    df_filled['城市'].fillna(df_filled['城市'].mode()[0], inplace=True)

# 对于可能随时间变化的列（如'上次消费金额'），有时用0填充更有业务意义
if '上次消费金额' in df_filled.columns:
    df_filled['上次消费金额'].fillna(0, inplace=True)

print("\n填充缺失值后，各列缺失值数量：")
print(df_filled.isnull().sum())

# 我们以'年收入'为例，假设它应该是一个合理的正值
if '年收入' in df_filled.columns:
    # 方法一：使用四分位距（IQR）法识别
    Q1 = df_filled['年收入'].quantile(0.25)
    Q3 = df_filled['年收入'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 找出异常值
    outliers = df_filled[(df_filled['年收入'] < lower_bound) | (df_filled['年收入'] > upper_bound)]
    print(f"\n使用IQR法发现的'年收入'异常值数量: {len(outliers)}")

    # 处理异常值：这里选择用上下边界值进行截断（Winsorization）
    df_filled['年收入'] = np.where(df_filled['年收入'] > upper_bound, upper_bound,
                                   np.where(df_filled['年收入'] < lower_bound, lower_bound, df_filled['年收入']))

    print("已对'年收入'的异常值进行截断处理。")