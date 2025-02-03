import numpy as np
import matplotlib.pyplot as plt

# 设置实验参数
num_actions = 2  # 2-armed bandit
num_steps = 1000  # 每次实验的步数
num_runs = 100  # 实验重复次数
alpha = 0.1  # 学习率
epsilon = 0.1  # ε-greedy 策略的探索率

# 真实的奖励均值
true_q_values = np.array([6, 7])

# 生成混合高斯奖励分布的函数
def get_reward(action):
    if action == 0:
        return np.random.normal(6, np.sqrt(15))  # 第一支臂的奖励
    else:
        if np.random.rand() < 0.5:
            return np.random.normal(11, np.sqrt(16))  # 第二支臂的奖励 (第一部分)
        else:
            return np.random.normal(3, np.sqrt(8))  # 第二支臂的奖励 (第二部分)

# 运行 ε-greedy Q-learning 算法
def run_epsilon_greedy():
    rewards = np.zeros((num_runs, num_steps))
    
    for run in range(num_runs):
        Q = np.zeros(num_actions)  # 初始化 Q 值为 0
        for step in range(num_steps):
            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)  # 以 ε 概率随机选择动作
            else:
                action = np.argmax(Q)  # 选择最优动作

            reward = get_reward(action)  # 获取奖励
            Q[action] = Q[action] + alpha * (reward - Q[action])  # 更新 Q 值
            rewards[run, step] = reward  # 记录奖励

    return np.mean(rewards, axis=0)  # 返回所有实验的平均累计奖励

# 运行梯度赌博机策略
def run_gradient_bandit():
    rewards = np.zeros((num_runs, num_steps))

    for run in range(num_runs):
        H = np.zeros(num_actions)  # 偏好值初始化为 0
        pi = np.ones(num_actions) / num_actions  # 初始均匀概率分布
        avg_reward = 0  # 记录平均奖励

        for step in range(num_steps):
            action = np.random.choice(num_actions, p=pi)  # 按概率选择动作
            reward = get_reward(action)  # 计算奖励
            avg_reward += (reward - avg_reward) / (step + 1)  # 更新平均奖励

            for a in range(num_actions):
                if a == action:
                    H[a] += alpha * (reward - avg_reward) * (1 - pi[a])
                else:
                    H[a] -= alpha * (reward - avg_reward) * pi[a]

            pi = np.exp(H) / np.sum(np.exp(H))  # 重新计算 softmax 概率
            rewards[run, step] = reward  # 记录奖励

    return np.mean(rewards, axis=0)  # 返回所有实验的平均累计奖励

# 运行实验
epsilon_greedy_results = run_epsilon_greedy()
gradient_bandit_results = run_gradient_bandit()

# 绘制比较曲线
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(epsilon_greedy_results) / np.arange(1, num_steps + 1), label="ε-Greedy (ε=0.1)", linestyle="--")
plt.plot(np.cumsum(gradient_bandit_results) / np.arange(1, num_steps + 1), label="Gradient Bandit (α=0.1)", linestyle="-")

plt.xlabel("Step/Time")
plt.ylabel("Average Accumulated Reward")
plt.title("Comparison of ε-Greedy and Gradient Bandit Policies")
plt.legend()
plt.grid()
plt.show()