import numpy as np


def epsilon_greedy(q_values, epsilon=0.1):
    """
    实现 ε-贪心策略
    Args:
        q_values: 一个数组，表示当前状态下每个动作的Q值估计。
        epsilon: 探索概率，介于0和1之间。
    Returns:
        selected_action: 根据策略选出的动作索引。
    """
    n_actions = len(q_values)

    # 以概率 epsilon 进行探索（随机选择）
    if np.random.random() < epsilon:
        selected_action = np.random.randint(n_actions)
    # 以概率 1-epsilon 进行开采（选择Q值最大的动作）
    else:
        # 如果多个动作Q值相同，随机选择一个
        selected_action = np.random.choice(np.where(q_values == np.max(q_values))[0])

    return selected_action


# 示例：假设在某个状态下，三个动作的Q值估计为 [1.5, 2.8, 2.3]
state_q_values = [1.5, 2.8, 2.3]
for i in range(10):
    action = epsilon_greedy(state_q_values, epsilon=0.2)
    print(f"第{i + 1}次选择: 动作 {action} (Q值: {state_q_values[action]:.1f})")