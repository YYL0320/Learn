# 1. 导入必要的库
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from scipy.stats import randint

# 2. 加载数据并划分
data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 * 0.8 = 0.2

# 3. 定义模型和参数分布
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': randint(50, 300),       # 树的数量
    'max_depth': randint(3, 20),            # 树的最大深度
    'min_samples_split': randint(2, 10),    # 内部节点分裂所需最小样本数
    'min_samples_leaf': randint(1, 5),      # 叶节点最小样本数
    'max_features': ['sqrt', 'log2']        # 寻找最佳分割时考虑的特征数
}

# 4. 执行随机搜索（带3折交叉验证）
random_search = RandomizedSearchCV(estimator=rf,
                                   param_distributions=param_dist,
                                   n_iter=30,          # 随机尝试30组参数
                                   cv=3,               # 3折交叉验证
                                   scoring='accuracy',
                                   random_state=42,
                                   verbose=1,
                                   n_jobs=-1)          # 使用所有CPU核心并行
random_search.fit(X_train_val, y_train_val)

# 5. 输出搜索结果
print("="*50)
print("随机搜索最佳参数：")
print(random_search.best_params_)
print(f"\n最佳交叉验证准确率：{random_search.best_score_:.4f}")

# 6. 在独立验证集上评估最佳模型
best_model = random_search.best_estimator_
y_val_pred = best_model.predict(X_val)
print("\n在验证集上的性能报告：")
print(classification_report(y_val, y_val_pred, target_names=data.target_names))

# 7. （最终步骤）用最佳参数在整个训练集上重新训练，并在测试集上评估
final_model = RandomForestClassifier(**random_search.best_params_, random_state=42)
final_model.fit(X_train, y_train) # 使用全部训练数据
y_test_pred = final_model.predict(X_test)
print("="*50)
print("最终模型在测试集（全新数据）上的性能报告：")
print(classification_report(y_test, y_test_pred, target_names=data.target_names))