# 导入必要的库
import pandas as pd
import numpy as np

# 加载数据
train_data = pd.read_csv('train.csv')  # 训练集，包含目标变量 Survived
test_data = pd.read_csv('test.csv')  # 测试集，不包含 Survived，用于最终评估

# 1. 初步查看数据
print("训练集形状：", train_data.shape)
print(train_data.info())  # 查看数据类型和缺失情况
print(train_data.head())  # 查看前几行数据

# 检查各列缺失值的数量
print(train_data.isnull().sum())


# ===================== 核心修改1：封装特征工程函数（确保训练/测试集处理一致） =====================
def preprocess_data(df, is_train=True, train_columns=None):
    df = df.copy()

    # 处理 Age（年龄）：中位数填充
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # 处理 Embarked（登船港口）：众数填充
    most_common_port = train_data['Embarked'].mode()[0]  # 始终用训练集的众数（避免数据泄露）
    df['Embarked'] = df['Embarked'].fillna(most_common_port)

    # 处理 Fare（船票价格）：中位数填充（训练集/测试集统一用训练集中位数）
    fare_median = train_data['Fare'].median()
    df['Fare'] = df['Fare'].fillna(fare_median)

    # 处理 Cabin（船舱）：直接删除
    df.drop(columns=['Cabin'], inplace=True, errors='ignore')

    # 将 Sex 列转换为数值：female -> 0, male -> 1
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

    # 将 Embarked 列转换为数值（独热编码）
    df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

    # 从 Name 列中提取称谓
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # 归类不常见的称谓
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs',
        'Master': 'Master', 'Dr': 'Rare', 'Rev': 'Rare',
        'Col': 'Rare', 'Major': 'Rare', 'Mlle': 'Miss',
        'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare',
        'Mme': 'Mrs', 'Capt': 'Rare', 'Sir': 'Rare'
    }
    df['Title'] = df['Title'].map(title_mapping)

    # 将 Title 列独热编码
    df = pd.get_dummies(df, columns=['Title'], prefix='Title')

    # 创建新特征：家庭规模
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 创建新特征：是否独自一人
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 删除不再需要的原始列
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch']
    df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

    # ===================== 核心修改2：特征对齐 =====================
    if not is_train:  # 测试集对齐到训练集的特征
        # 1. 补充测试集缺失的列，值为0
        missing_cols = set(train_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        # 2. 删除测试集多出来的列
        extra_cols = set(df.columns) - set(train_columns)
        df.drop(columns=extra_cols, inplace=True, errors='ignore')
        # 3. 按训练集列顺序排序
        df = df[train_columns]

    return df


# 处理训练集
train_processed = preprocess_data(train_data, is_train=True)
# 保存训练集的特征列（用于测试集对齐）
train_feature_columns = train_processed.drop('Survived', axis=1).columns.tolist()

# 处理测试集（关键：传入训练集的特征列进行对齐）
test_passenger_ids = test_data['PassengerId']  # 保存测试集ID用于提交
test_processed = preprocess_data(test_data, is_train=False, train_columns=train_feature_columns)

print("特征工程后的训练集列名：", train_processed.columns.tolist())
print("特征工程后的测试集列名：", test_processed.columns.tolist())

# 导入机器学习库
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 准备数据
X = train_processed.drop('Survived', axis=1)
y = train_processed['Survived']

# 分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林分类器
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 在验证集上进行预测
y_pred = model.predict(X_val)

# 评估模型准确率
accuracy = accuracy_score(y_val, y_pred)
print(f"模型在验证集上的准确率为：{accuracy:.4f} (即 {accuracy * 100:.2f}%)")

# 尝试不同的最大深度
for depth in [3, 5, 10, None]:  # None 表示不限制深度
    model_temp = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    model_temp.fit(X_train, y_train)
    y_pred_temp = model_temp.predict(X_val)
    acc = accuracy_score(y_val, y_pred_temp)
    print(f"max_depth={depth} 时，验证集准确率：{acc:.4f}")

# 获取特征重要性
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("特征重要性排名：")
print(feature_importances)

# 使用全部训练数据重新训练最终模型
final_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
final_model.fit(X, y)  # 这次使用全部训练数据 X, y

# ===================== 核心修改3：预测时使用对齐后的测试集 =====================
final_predictions = final_model.predict(test_processed)

# 创建提交文件
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': final_predictions
})
submission.to_csv('my_titanic_submission.csv', index=False)
print("预测结果已保存至 'my_titanic_submission.csv'，可以提交到Kaggle平台查看排名！")