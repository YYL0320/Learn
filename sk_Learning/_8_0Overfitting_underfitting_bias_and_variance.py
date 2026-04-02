import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

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
# 设置图表样式
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
# -------------------------- 设置中文字体 end --------------------------

# 加载数据
data = load_diabetes()
X, y = data.data, data.target
# 只使用一个特征（更适合多项式回归演示）
X = X[:, np.newaxis, 2]  # 选择第三个特征（BMI）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 定义学习曲线绘制函数（优化版）
def plot_learning_curve(estimator, title, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    绘制学习曲线
    参数：
        estimator: 模型估计器
        title: 图表标题
        X: 特征数据
        y: 目标变量
        cv: 交叉验证折数
        train_sizes: 训练样本比例
    """
    # 获取学习曲线数据
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='neg_mean_squared_error',
        train_sizes=train_sizes, random_state=42, n_jobs=-1
    )

    # 计算均值和标准差
    train_scores_mean = -train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)
    test_scores_std = test_scores.std(axis=1)

    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes_abs,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color='r')
    plt.fill_between(train_sizes_abs,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color='g')

    # 绘制均值曲线
    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='r', linewidth=2,
             markersize=8, label='训练集 MSE')
    plt.plot(train_sizes_abs, test_scores_mean, 'o-', color='g', linewidth=2,
             markersize=8, label='验证集 MSE')

    # 设置图表属性
    plt.xlabel('训练样本数量', fontsize=12)
    plt.ylabel('均方误差 (MSE)', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()

    # 打印模型在测试集上的表现
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{title} - 测试集 MSE: {mse:.2f}")


# 1. 欠拟合模型（1阶多项式 - 线性回归）
print("=" * 60)
print("欠拟合模型（1阶多项式 - 线性回归）")
print("=" * 60)
plot_learning_curve(
    make_pipeline(StandardScaler(), PolynomialFeatures(1), LinearRegression()),
    '欠拟合模型学习曲线（1阶多项式）',
    X, y
)

# 2. 良好拟合模型（2阶多项式）
print("\n" + "=" * 60)
print("良好拟合模型（2阶多项式）")
print("=" * 60)
plot_learning_curve(
    make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()),
    '良好拟合模型学习曲线（2阶多项式）',
    X, y
)

# 3. 过拟合模型（8阶多项式）
print("\n" + "=" * 60)
print("过拟合模型（8阶多项式）")
print("=" * 60)
plot_learning_curve(
    make_pipeline(StandardScaler(), PolynomialFeatures(8), LinearRegression()),
    '过拟合模型学习曲线（8阶多项式）',
    X, y
)

# 额外：可视化不同阶数模型的拟合效果
plt.figure(figsize=(12, 8))
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# 绘制原始数据点
plt.scatter(X_train, y_train, alpha=0.5, label='训练数据', color='blue', s=30)
plt.scatter(X_test, y_test, alpha=0.5, label='测试数据', color='orange', s=30)

# 绘制不同阶数的拟合曲线
orders = [1, 2, 8]
colors = ['red', 'green', 'purple']
labels = ['1阶（欠拟合）', '2阶（良好拟合）', '8阶（过拟合）']

for i, order in enumerate(orders):
    model = make_pipeline(StandardScaler(), PolynomialFeatures(order), LinearRegression())
    model.fit(X_train, y_train)
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, color=colors[i], linewidth=2, label=labels[i])

plt.xlabel('BMI 特征（标准化）', fontsize=12)
plt.ylabel('糖尿病进展指标', fontsize=12)
plt.title('不同阶数多项式回归的拟合效果对比', fontsize=14, pad=20)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()