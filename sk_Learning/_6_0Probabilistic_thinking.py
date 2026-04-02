# 定义已知概率
P_disease = 0.001          # P(病)
P_positive_given_disease = 0.99   # P(阳|病)
P_negative_given_healthy = 0.99   # P(阴|健康)

# 计算派生概率
P_healthy = 1 - P_disease          # P(健康)
P_positive_given_healthy = 1 - P_negative_given_healthy  # P(阳|健康) = 1 - 特异度

# 计算全概率 P(阳)
# P(阳) = P(阳|病)*P(病) + P(阳|健康)*P(健康)
P_positive = (P_positive_given_disease * P_disease) + (P_positive_given_healthy * P_healthy)

# 应用贝叶斯定理计算 P(病|阳)
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print(f"即使检测为阳性，真正患病的后验概率 P(病|阳) 仅为: {P_disease_given_positive:.2%}")