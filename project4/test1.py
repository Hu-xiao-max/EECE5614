import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 固定随机种子，便于调试
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#########################################
# 迷宫环境：定义一个 8x8 的迷宫示例
#########################################
class MazeEnv:
    def __init__(self):
        # 迷宫说明：
        # 数值含义：
        # 0 - 普通空白格（reward = -1）
        # 1 - 墙壁（不可进入，若试图移动到墙壁则原地并给 -1.8 作为惩罚）
        # 2 - 目标格（goal，reward = +100）
        # 3 - 红色区域（额外-10，故总reward = -11，即 -1 -10）
        # 4 - 黄色区域（额外-5，故总reward = -6，即 -1 -5）
        self.maze = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 4, 0, 0, 0, 2, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 3, 1, 1, 1, 1, 4, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 4, 0, 0, 0, 1, 4, 0, 1],
            [1, 1, 1, 1, 3, 0, 1, 0, 0, 1],
            [1, 0, 5, 3, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 3, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # 目标格在右下角
        ])
        self.n_rows, self.n_cols = self.maze.shape

        self.P = 0.025  # 扰动概率
        self.max_steps = 50  # 每集最大步数

        # 动作映射：0: 上, 1: 右, 2: 下, 3: 左
        self.actions = {0: (-1, 0),
                        1: (0, 1),
                        2: (1, 0),
                        3: (0, -1)}
        # 对于垂直与水平方向，确定垂直（perpendicular）的动作
        self.perp = {
            0: [1, 3],  # 若原本向上，则垂直方向为左右
            1: [0, 2],  # 向右则垂直为上下
            2: [1, 3],
            3: [0, 2]
        }

    def reset(self):
        # 随机选择一个非目标且非墙壁的位置作为初始状态
        free_cells = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.maze[i, j] != 1 and self.maze[i, j] != 2:
                    free_cells.append((i, j))
        self.state = random.choice(free_cells)
        self.steps = 0
        return np.array([self.state[1], self.state[0]], dtype=np.float32)  # 使用 (x,y) 顺序：x=列, y=行

    def step(self, action):
        """
        根据给定 action 以及扰动概率 P 返回下一个状态、奖励以及是否结束
        """
        self.steps += 1
        # 根据 epsilon 采样确定实际动作（考虑扰动转移）
        rand_val = np.random.rand()
        if rand_val < (1 - self.P):
            chosen_action = action
        else:
            # 随机在垂直方向两个动作中选一个
            chosen_action = random.choice(self.perp[action])
        
        # 计算预期下一个位置
        move = self.actions[chosen_action]
        next_row = self.state[0] + move[0]
        next_col = self.state[1] + move[1]
        
        # 检查边界
        if next_row < 0 or next_row >= self.n_rows or next_col < 0 or next_col >= self.n_cols:
            # 出界当作撞墙处理
            reward = -1.8  
            next_state = self.state  # 原地停留
        # 检查是否撞墙
        elif self.maze[next_row, next_col] == 1:
            reward = -1.8
            next_state = self.state  # 不动
        else:
            # 非墙且在边界内，获得 cell 对应的 reward
            cell_val = self.maze[next_row, next_col]
            if cell_val == 0:
                reward = -1
            elif cell_val == 3:
                reward = -11  # -1 - 10
            elif cell_val == 4:
                reward = -6   # -1 - 5
            elif cell_val == 2:
                reward = 100
            else:
                reward = -1

            next_state = (next_row, next_col)
        
        # 更新状态
        self.state = next_state

        # 判断是否达到终点或步数达到上限
        done = False
        if self.maze[self.state[0], self.state[1]] == 2:
            done = True
        if self.steps >= self.max_steps:
            done = True
        
        # 返回状态以 (x,y) 顺序
        return np.array([self.state[1], self.state[0]], dtype=np.float32), reward, done

    def render(self):
        # 简单的文本渲染迷宫状态
        maze_copy = self.maze.copy().astype(str)
        r, c = self.state
        maze_copy[r, c] = 'A'
        print("\n".join([" ".join(row) for row in maze_copy]))
    

#########################################
# 经验回放缓冲区
#########################################
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


#########################################
# Q-Network（标准与 Double DQN 使用相同网络结构）
#########################################
class QNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=4, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


#########################################
# Dueling Q-Network
#########################################
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=4, hidden_dim=64):
        super(DuelingQNetwork, self).__init__()
        # 共享的特征提取层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # 分支1：状态价值 V(s)
        self.fc_value = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        # 分支2：优势函数 A(s,a)
        self.fc_adv = nn.Linear(hidden_dim, hidden_dim)
        self.advantage = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # Value stream
        v = torch.relu(self.fc_value(x))
        v = self.value(v)
        # Advantage stream
        a = torch.relu(self.fc_adv(x))
        a = self.advantage(a)
        # Combine：减去均值以消除冗余
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


#########################################
# DQN Agent，支持标准 DQN、Double DQN 以及 Dueling DQN
#########################################
class DQNAgent:
    def __init__(self, env, variant='DQN', replay_capacity=5000, batch_size=64, gamma=0.99,
                 lr=1e-3, target_update_eta=1e-3, update_freq=5,
                 epsilon_decay=0.995, epsilon_min=0.1, initial_epsilon=1.0):
        """
        variant: 'DQN' / 'DoubleDQN' / 'DuelingDQN'
        update_freq: 每 N_QU 步进行一次网络更新
        """
        self.env = env
        self.variant = variant
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.target_update_eta = target_update_eta
        self.update_freq = update_freq

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # 根据 variant 初始化相应的网络结构
        if variant == 'DuelingDQN':
            self.qnet = DuelingQNetwork()
            self.target_net = DuelingQNetwork()
        else:
            self.qnet = QNetwork()
            self.target_net = QNetwork()

        # 将 target 网络初始化为与 qnet 相同的参数
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.num_updates = 0

    def select_action(self, state):
        """
        epsilon-greedy 策略选取动作，state 为 numpy 数组（shape: (2,)）
        """
        if random.random() < self.epsilon:
            return random.randrange(4)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 变成 (1,2) 的 tensor
            with torch.no_grad():
                q_values = self.qnet(state_tensor)
            return q_values.argmax().item()

    def soft_update(self):
        """采用软更新方式更新 target 网络"""
        for target_param, param in zip(self.target_net.parameters(), self.qnet.parameters()):
            target_param.data.copy_(self.target_update_eta * param.data + (1 - self.target_update_eta) * target_param.data)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0  # 更新未发生时 loss 为 0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)  # shape: (B,1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # 当前 Q 值
        curr_Q = self.qnet(states).gather(1, actions)

        with torch.no_grad():
            if self.variant == 'DoubleDQN':
                # 使用 qnet 选取下一个动作，再用 target_net 评估
                next_q_values = self.qnet(next_states)
                next_actions = next_q_values.argmax(dim=1, keepdim=True)
                next_Q = self.target_net(next_states).gather(1, next_actions)
                target_Q = rewards + self.gamma * next_Q * (1 - dones)
            else:
                # 标准 DQN 或 Dueling DQN（结构仅不同于网络模型，此处目标计算相同）
                next_Q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
                target_Q = rewards + self.gamma * next_Q * (1 - dones)

        loss = self.criterion(curr_Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 target 网络（软更新）
        self.soft_update()
        self.num_updates += 1

        return loss.item()

    def train(self, num_episodes=500):
        episode_rewards = []
        episode_losses = []
        total_steps = 0

        for epi in range(1, num_episodes + 1):
            state = self.env.reset()
            epi_reward = 0
            epi_loss = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                epi_reward += reward
                total_steps += 1

                # 每 update_freq 步更新一次网络
                if total_steps % self.update_freq == 0:
                    loss = self.update()
                    epi_loss += loss

            # 衰减 epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            episode_rewards.append(epi_reward)
            episode_losses.append(epi_loss)

            if epi % 10 == 0:
                print(f"Episode {epi}, Reward: {epi_reward:.2f}, Loss: {epi_loss:.4f}, Epsilon: {self.epsilon:.4f}")
        return episode_rewards, episode_losses

    def get_policy(self):
        """
        计算迷宫中每个状态最优动作（对于非墙格）
        返回形状 (n_rows, n_cols) 的动作矩阵（若为墙则标记为 -1）
        """
        policy = -np.ones((self.env.n_rows, self.env.n_cols), dtype=int)
        for i in range(self.env.n_rows):
            for j in range(self.env.n_cols):
                if self.env.maze[i, j] != 1:  # 非墙格
                    state = np.array([j, i], dtype=np.float32)
                    with torch.no_grad():
                        q_vals = self.qnet(torch.FloatTensor(state).unsqueeze(0))
                    policy[i, j] = q_vals.argmax().item()
        return policy

    def get_value_function(self):
        """
        计算迷宫中每个状态的最大 Q 值（即状态价值）
        返回形状 (n_rows, n_cols) 的值矩阵（墙格设为 NaN）
        """
        value_func = np.zeros((self.env.n_rows, self.env.n_cols))
        for i in range(self.env.n_rows):
            for j in range(self.env.n_cols):
                if self.env.maze[i, j] != 1:
                    state = np.array([j, i], dtype=np.float32)
                    with torch.no_grad():
                        q_vals = self.qnet(torch.FloatTensor(state).unsqueeze(0))
                    value_func[i, j] = q_vals.max().item()
                else:
                    value_func[i, j] = np.nan
        return value_func

    def get_path(self, start_state=None):
        """
        从起点开始，依据贪婪策略生成一条路径，直至到达目标或达到步数上限
        """
        if start_state is None:
            state = self.env.reset()
        else:
            state = start_state
        path = []
        path.append(state)
        done = False
        steps = 0
        while not done and steps < self.env.max_steps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.qnet(state_tensor)
            action = q_vals.argmax().item()
            next_state, reward, done = self.env.step(action)
            path.append(next_state)
            state = next_state
            steps += 1
        return path


#########################################
# 绘图工具：绘制移动平均曲线、策略（用箭头表示）、状态值以及路径
#########################################
def moving_average(data, window=25):
    ma = np.convolve(data, np.ones(window)/window, mode='valid')
    return ma

def plot_training(episode_rewards, episode_losses):
    episodes = np.arange(1, len(episode_rewards)+1)
    avg_rewards = moving_average(episode_rewards)
    avg_losses = moving_average(episode_losses)
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(episodes[len(episodes)-len(avg_rewards):], avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (window=25)")
    plt.title("Training: Average Reward")
    
    plt.subplot(1,2,2)
    plt.plot(episodes[len(episodes)-len(avg_losses):], avg_losses)
    plt.xlabel("Episode")
    plt.ylabel("Average Loss (window=25)")
    plt.title("Training: Average Loss")
    
    plt.tight_layout()
    plt.show()

def plot_policy(policy, env):
    """
    绘制策略，每个格子用箭头标出动作，并填充颜色块，背景颜色根据迷宫环境中的格子类型设置：
    0 - 普通空白格（white）
    1 - 墙壁（black）
    2 - 目标格（gold）
    3 - 红色区域（red）
    4 - 黄色区域（yellow）
    5 - 蓝色区域（blue）
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    
    # 定义每个动作对应的箭头符号
    action_arrows = {0: '↑', 1: '→', 2: '↓', 3: '←', -1: '■'}
    
    # 使用 env.maze 作为背景颜色块
    maze = env.maze.copy()
    colors = ['white', 'black', 'green', 'red', 'yellow', 'blue']
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    plt.figure()
    plt.imshow(maze, cmap=cmap, norm=norm, origin='upper')
    
    # 在每个格子上标记对应动作的箭头
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            symbol = action_arrows.get(policy[i, j], '?')
            plt.text(j, i, symbol, ha='center', va='center', fontsize=14, color='black')
    
    plt.title("Final Policy (Arrow indicates best action)")
    plt.axis('off')
    plt.show()

def plot_value_function(value_func, env):
    """
    绘制状态价值函数（最大 Q 值），背景采用环境地图的色块，
    并在每个非墙格内标注状态价值：
    
      0 - 普通空白格 (white)
      1 - 墙壁 (black)
      2 - 目标格 (gold)
      3 - 红色区域 (red)
    4 - 黄色区域 (yellow)
    5 - 蓝色区域 (blue)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import numpy as np

    # 定义颜色映射，与迷宫数值对应
    colors = ['white', 'black', 'green', 'red', 'yellow', 'blue']
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6, 6))
    # 根据 env.maze 的数值生成背景颜色块
    plt.imshow(env.maze, cmap=cmap, norm=norm, origin='upper')
    plt.title("State Value Function (max Q-value)")

    # 在每个非墙格（maze != 1）上标注状态价值
    for i in range(env.n_rows):
        for j in range(env.n_cols):
            if env.maze[i, j] != 1:
                val = value_func[i, j]
                if not np.isnan(val):
                    plt.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=12, color="black")

    plt.axis('off')
    plt.show()

def plot_path(path, env):
    """
    绘制路径，背景为迷宫地图。地图块带有颜色，分别代表不同类型区域：
    0 - 普通空白格（白色）
    1 - 墙壁（黑色）
    2 - 目标格（gold）
    3 - 红色区域（red）
    4 - 黄色区域（yellow）
    5 - 蓝色区域（blue）
    使用箭头标示移动方向，并用散点标记起点与终点
    """
    from matplotlib.colors import ListedColormap
    
    maze = env.maze.copy()
    plt.figure()
    # 定义颜色映射，顺序对应迷宫中数值的意义
    cmap = ListedColormap(['white', 'black', 'green', 'red', 'yellow', 'blue'])
    plt.imshow(maze, cmap=cmap, origin='upper')
    
    path = np.array(path)
    # 使用箭头展示路径中每一步的移动
    for i in range(len(path) - 1):
        x, y = path[i]
        x_next, y_next = path[i + 1]
        dx = x_next - x
        dy = y_next - y
        plt.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, fc='blue', ec='blue')
    
    # 标记起点和终点
    plt.scatter(path[0, 0], path[0, 1], c='green', marker='o', s=100, label='Start')
    plt.scatter(path[-1, 0], path[-1, 1], c='red', marker='*', s=100, label='Goal')
    
    plt.title("Path Following the Obtained Policy")
    plt.legend()
    plt.show()


#########################################
# 主函数
#########################################
def main():
    # 选择 variant: 'DQN', 'DoubleDQN', 'DuelingDQN'
    variant = 'DQN'  # 可改为 'DQN' 或 'DoubleDQN'
    
    print(f"Training variant: {variant}")
    
    # 创建环境与 agent
    env = MazeEnv()
    agent = DQNAgent(env, variant=variant, replay_capacity=5000, batch_size=64,
                     gamma=0.99, lr=1e-3, target_update_eta=1e-3, update_freq=5,
                     epsilon_decay=0.995, epsilon_min=0.1, initial_epsilon=1.0)
    
    num_episodes = 30000
    episode_rewards, episode_losses = agent.train(num_episodes=num_episodes)
    
    # 绘制训练曲线
    plot_training(episode_rewards, episode_losses)

    # 得到最终策略与状态价值
    policy = agent.get_policy()
    value_func = agent.get_value_function()
    plot_policy(policy, env)
    plot_value_function(value_func, env)

    # 从一个初始状态出发，获得一条路径
    path = agent.get_path()
    plot_path(path, env)

if __name__ == '__main__':
    main()