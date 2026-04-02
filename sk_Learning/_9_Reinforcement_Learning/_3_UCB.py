import numpy as np
import math


class UCB:
    def __init__(self, n_actions, c=2):
        self.n_actions = n_actions
        self.c = c  # 探索参数
        self.Q = np.zeros(n_actions)  # 动作价值估计
        self.N = np.zeros(n_actions)  # 动作选择次数
        self.total_steps = 0

    def select_action(self):
        self.total_steps += 1
        # 确保每个动作至少被选择一次
        if np.any(self.N == 0):
            action = np.random.choice(np.where(self.N == 0)[0])
        else:
            # 计算每个动作的UCB值： Q(a) + c * sqrt(ln(t) / N(a))
            ucb_values = self.Q + self.c * np.sqrt(np.log(self.total_steps) / self.N)
            action = np.argmax(ucb_values)
        return action

    def update(self, action, reward):
        """更新动作的价值估计"""
        self.N[action] += 1
        # 增量更新Q值: NewEstimate = OldEstimate + (1/N) * (Target - OldEstimate)
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


# 模拟一个多臂老虎机问题，每个臂的真实奖励概率不同
true_means = [0.1, 0.5, 0.9]  # 三个臂的真实平均奖励
n_actions = len(true_means)
bandit = UCB(n_actions, c=2)

total_reward = 0
for step in range(1000):
    action = bandit.select_action()
    # 模拟拉动老虎机臂，以一定概率获得奖励1，否则为0
    reward = 1 if np.random.random() < true_means[action] else 0
    bandit.update(action, reward)
    total_reward += reward

print(f"UCB策略在1000步后获得的总奖励: {total_reward}")
print(f"各动作被选择的次数: {bandit.N}")
print(f"各动作的Q值估计: {bandit.Q}")