import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import gym
from gym import spaces
from collections import namedtuple, deque

# 定义GCN层


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = self.fc(out)
        return F.relu(out)

# 修改后的DQN模型，加入GCN层


class DQNWithSpatial(nn.Module):
    def __init__(self, state_size, action_size, gcn_hidden_size):
        super(DQNWithSpatial, self).__init__()
        self.gcn1 = GCNLayer(state_size, gcn_hidden_size)
        # 这里将输入调整为 (gcn_hidden_size * num_nodes)
        self.fc1 = nn.Linear(gcn_hidden_size * state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x, adj):
        x = self.gcn1(x.transpose(0, 1), adj)  # 转置x，使其与邻接矩阵匹配
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义ReplayMemory，用于存储经验
Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 修改的环境类，加入GCN处理


class CellSelectionEnvWithLSTMAndGCN(gym.Env):
    def __init__(self, data_matrix, error_bound, quality_threshold, lstm_model, seq_length, adj):
        super(CellSelectionEnvWithLSTMAndGCN, self).__init__()
        self.data_matrix = data_matrix
        self.num_cells = data_matrix.shape[0]
        self.num_hours = data_matrix.shape[1]
        self.error_bound = error_bound
        self.quality_threshold = quality_threshold
        self.lstm_model = lstm_model
        self.seq_length = seq_length
        self.selected_cells = []
        self.current_time = 0
        self.adj = torch.tensor(adj, dtype=torch.float32)

        self.action_space = spaces.Discrete(self.num_cells)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_cells,), dtype=np.float32)

        # 将 LSTM 输出的 (10, 64) 转换为 (10, 36)
        self.lstm_to_gcn = nn.Linear(64, self.num_cells)

    def reset(self):
        self.selected_cells = []
        self.current_seq = np.zeros((self.seq_length, self.num_cells))
        self.current_time = 0
        return self._get_state()

    def _get_state(self):
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_output, _ = self.lstm_model(torch.tensor(
                self.current_seq, dtype=torch.float32).unsqueeze(0))
            lstm_output = lstm_output.squeeze(0)  # (10, 64)

            # 使用线性变换将 LSTM 输出转换为 GCN 的输入
            lstm_output = self.lstm_to_gcn(lstm_output)  # (10, 36)

            gcn_state = lstm_output.matmul(self.adj)  # (10, 36) x (36, 36)
            return gcn_state.numpy()

    def step(self, action):
        if action not in self.selected_cells:
            self.selected_cells.append(action)

        new_data = self.data_matrix[:, self.current_time]
        self.current_seq = np.roll(self.current_seq, -1, axis=0)
        self.current_seq[-1] = new_data

        state = self._get_state()
        reward = self._calculate_reward()
        done = self._check_done()

        self.current_time += 1

        return state, reward, done, {}

    def _calculate_reward(self):
        # 根据选中的单元格计算奖励
        return 1.0  # 简化奖励函数，实际应用中应根据特定任务定义

    def _check_done(self):
        return self.current_time >= self.num_hours

# 训练过程使用新的DQN模型


def train_dqn_with_gcn(env, num_episodes, gamma, epsilon, lr, memory_size=10000, batch_size=64, target_update=10):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    gcn_hidden_size = 36  # 将隐藏层大小设置为36以匹配 LSTM 输出

    policy_net = DQNWithSpatial(state_size, action_size, gcn_hidden_size)
    target_net = DQNWithSpatial(state_size, action_size, gcn_hidden_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    memory = ReplayMemory(memory_size)

    rewards = []
    selected_cells_ratios = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = []

        for t in range(env.num_hours):
            state_tensor = torch.tensor(
                state, dtype=torch.float32).unsqueeze(0)

            if random.random() < epsilon:
                action = random.choice(range(action_size))
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor, env.adj)
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            memory.push(state, action, reward, next_state)

            state = next_state
            episode_rewards.append(reward)

            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state = zip(
                    *transitions)

                batch_state = torch.tensor(batch_state, dtype=torch.float32)
                batch_action = torch.tensor(
                    batch_action, dtype=torch.long).unsqueeze(1)
                batch_reward = torch.tensor(
                    batch_reward, dtype=torch.float32).unsqueeze(1)
                batch_next_state = torch.tensor(
                    batch_next_state, dtype=torch.float32)

                current_q_values = policy_net(
                    batch_state, env.adj).gather(1, batch_action)
                next_q_values = target_net(batch_next_state, env.adj).max(1)[
                    0].detach().unsqueeze(1)
                expected_q_values = batch_reward + (gamma * next_q_values)

                loss = criterion(current_q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done or (t % target_update == 0):
                target_net.load_state_dict(policy_net.state_dict())

            selected_cells_ratio = len(env.selected_cells) / env.num_cells
            selected_cells_ratios.append(selected_cells_ratio)

            if done:
                break
        print(f'Episode {episode}, Selected Cells Ratio: {
              len(env.selected_cells)} / {env.num_cells}')
        episode_total_reward = np.sum(episode_rewards)
        rewards.append(episode_total_reward)

        if episode % 10 == 0:
            print(f'Episode {episode}, Loss: {loss.item()}')

    return policy_net, rewards, selected_cells_ratios


# 模拟数据和参数（示例）
num_cells = 36
num_hours = 24
data_matrix = np.random.rand(num_cells, num_hours)
error_bound = 0.1
quality_threshold = 0.8
seq_length = 10
adj = np.eye(num_cells)  # 使用单位矩阵作为邻接矩阵的示例
lstm_model = nn.LSTM(input_size=num_cells, hidden_size=64,
                     num_layers=1)  # 简单的LSTM模型
gamma = 0.99
epsilon = 0.1
lr = 0.001
num_episodes = 100

# 使用新的环境初始化和训练
env_with_gcn = CellSelectionEnvWithLSTMAndGCN(
    data_matrix, error_bound, quality_threshold, lstm_model, seq_length, adj)

policy_net_with_gcn, train_rewards_with_gcn, train_selected_cells_ratios_with_gcn = train_dqn_with_gcn(
    env_with_gcn, num_episodes, gamma, epsilon, lr)

print(f'Average Training Reward with GCN: {np.mean(train_rewards_with_gcn)}')
average_ratio_with_gcn = sum(
    train_selected_cells_ratios_with_gcn) / len(train_selected_cells_ratios_with_gcn)
print(f'Average Select Ratio with GCN: {average_ratio_with_gcn:.2f}')
print(f'Average Select Cell with GCN: {(36 * average_ratio_with_gcn):.1f}')
