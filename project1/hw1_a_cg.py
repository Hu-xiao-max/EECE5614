import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. 定义拉杆奖励的采样函数
# =============================
def sample_reward(action):
    """
    对应题目设定:
    - action=0: 服从 N(6, 15)
    - action=1: 以0.5概率来自 N(11,16)，以0.5概率来自 N(3,8)
    返回一次采样得到的奖励
    """
    if action == 0:
        # 拉杆 a1: mean=6, var=15 -> std = sqrt(15)
        return np.random.normal(6, np.sqrt(15))
    else:
        # 拉杆 a2: 50% 来自 N(11,16), 50% 来自 N(3,8)
        if np.random.rand() < 0.5:
            return np.random.normal(11, np.sqrt(16))
        else:
            return np.random.normal(3, np.sqrt(8))


# =============================
# 2. 定义学习率函数
# =============================
def alpha_constant(k):
    # alpha = 1, 与 k 无关
    return 1.0

def alpha_decay_09(k):
    # alpha = 0.9^k, 注意 k 从 1 开始，故 k=1 -> 0.9
    return 0.9**k

def alpha_inv_ln(k):
    # alpha = 1 / [1 + ln(1 + k)]
    return 1.0/(1.0 + np.log(1.0 + k))

def alpha_inv_k(k):
    # alpha = 1 / k
    # 注意 k=0 时要特殊处理，这里从1开始算即可
    return 1.0 / k if k>0 else 1.0


# 将四种学习率策略封装在一个列表中，便于迭代
alpha_strategies = [
    ("alpha=1"          , alpha_constant),
    ("alpha=0.9^k"      , alpha_decay_09),
    ("alpha=1/(1+ln(1+k))", alpha_inv_ln),
    ("alpha=1/k"        , alpha_inv_k),
]

# =============================
# 3. 定义 epsilon-greedy 策略
# =============================
def choose_action(Q, epsilon):
    """
    Q: 当前 Q值列表，比如 [Q(a1), Q(a2)]
    epsilon: 探索率
    返回动作 0 或 1
    """
    if np.random.rand() < epsilon:
        # 随机动作
        return np.random.choice([0, 1])
    else:
        # 贪心动作
        return np.argmax(Q)


# =============================
# 4. 运行实验并作图
# =============================
def run_bandit(alpha_func, epsilon, num_steps=1000, num_runs=100):
    """
    针对给定的 alpha 策略函数和 epsilon，
    重复num_runs次，每次跑num_steps步，返回：
      - average_accum_rewards: 大小(num_steps,)的数组，
        表示在每个step处的平均累积奖励
      - final_Q: 在所有run结束后，对应最后一步 Q(a1),Q(a2) 的平均值
    """
    # 用来累加每一步的累计奖励（为了后面做平均）
    all_runs_accR = np.zeros((num_runs, num_steps))
    
    # 记录所有运行结束后的 Q值, 以便做平均
    final_Q_vals = np.zeros((num_runs, 2))
    
    for i in range(num_runs):
        # 初始化 Q(a1)=Q(a2)=0
        Q = np.array([0.0, 0.0])
        cumulative_reward = 0.0
        
        for k in range(num_steps):
            # 选择动作
            action = choose_action(Q, epsilon)
            # 采样奖励
            reward = sample_reward(action)
            # 更新累计奖励
            cumulative_reward += reward
            # 计算此时的学习率
            alpha = alpha_func(k+1)  # k+1是当前(1-based)步数
            # 更新 Q 值
            Q[action] += alpha * (reward - Q[action])
            
            # 记录到当前步为止的平均累计奖励
            all_runs_accR[i, k] = cumulative_reward / (k+1)
        
        # 单次run结束，记录最终的 Q(a1), Q(a2)
        final_Q_vals[i, :] = Q
    
    # 计算 100 次独立运行在每一步的平均累积奖励
    average_accum_rewards = np.mean(all_runs_accR, axis=0)
    # 计算 100 次独立运行最后的平均 Q 值
    final_Q = np.mean(final_Q_vals, axis=0)
    
    return average_accum_rewards, final_Q


# 主程序：对每种 alpha 策略画一张图，图里包含不同 epsilon 的曲线
epsilons = [0.0, 0.1, 0.2, 0.5]

plt.figure(figsize=(12,10))

for idx, (alpha_name, alpha_func) in enumerate(alpha_strategies, start=1):
    plt.subplot(2,2,idx)  # 2x2 子图
    for eps in epsilons:
        avg_accR, finalQ = run_bandit(alpha_func, eps, num_steps=1000, num_runs=100)
        plt.plot(avg_accR, label=f"eps={eps}")
    
    plt.title(alpha_name)
    plt.xlabel("Steps")
    plt.ylabel("Average Accumulated Reward")
    plt.legend()

plt.tight_layout()
plt.show()

# 如果还需要打印每个 (alpha, eps) 对应 1000 步后的平均 Q(a1), Q(a2)，
# 可以再做一个循环逐一输出：
print("===== Final Q-values after 1000 steps (averaged over 100 runs) =====")
for alpha_name, alpha_func in alpha_strategies:
    for eps in epsilons:
        _, finalQ = run_bandit(alpha_func, eps, num_steps=1000, num_runs=100)
        print(f"{alpha_name}, eps={eps}: Q(a1)={finalQ[0]:.3f}, Q(a2)={finalQ[1]:.3f}")