import torch
import torch.optim as optim
import torch.nn as nn
import random
from gym import spaces
import gym
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor
import matplotlib.pyplot as plt
from collections import deque

# 读取数据
data = pd.read_csv('../PM25.csv', header=None).values

# 将数据转换成36x264的矩阵
data_matrix = data.reshape((36, 264))

# 数据标准化
scaler = MinMaxScaler()
data_matrix = scaler.fit_transform(data_matrix.T).T

# 定义LSTM模型


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 创建序列数据


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 48  # 使用过去48小时的数据预测下一个时间点
train_x, train_y = create_sequences(data_matrix.T, seq_length)
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)

# 定义并训练LSTM模型
input_size = data_matrix.shape[0]
hidden_size = 64
num_layers = 2
output_size = data_matrix.shape[0]

lstm_model = LSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
num_epochs = 100

for epoch in range(num_epochs):
    lstm_model.train()
    optimizer.zero_grad()
    output = lstm_model(train_x)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 定义多智能体环境


class MultiAgentCellSelectionEnv(gym.Env):
    def __init__(self, data_matrix, error_bound, quality_threshold, lstm_model, seq_length, num_agents):
        super(MultiAgentCellSelectionEnv, self).__init__()
        self.data_matrix = data_matrix
        self.num_cells = data_matrix.shape[0]
        self.num_hours = data_matrix.shape[1]
        self.error_bound = error_bound
        self.quality_threshold = quality_threshold
        self.lstm_model = lstm_model
        self.seq_length = seq_length
        self.num_agents = num_agents
        self.current_time = 0

        # 为每个智能体维护独立的selected_cells列表
        self.selected_cells = [[] for _ in range(num_agents)]

        # 动作空间：每个智能体选择一个小区
        self.action_space = spaces.MultiDiscrete([self.num_cells] * num_agents)

        # 状态空间：LSTM输出的状态表示
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_agents, self.num_cells), dtype=np.float32)

    def reset(self):
        self.selected_cells = [[] for _ in range(self.num_agents)]
        self.current_seq = np.zeros(
            (self.num_agents, self.seq_length, self.num_cells))
        self.current_time = 0
        return self._get_state()

    def _get_state(self):
        self.lstm_model.eval()
        with torch.no_grad():
            state = []
            for i in range(self.num_agents):
                state.append(self.lstm_model(torch.tensor(
                    self.current_seq[i], dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy())
            return np.array(state)

    def step(self, actions):
        for i, action in enumerate(actions):
            if action not in self.selected_cells[i]:
                self.selected_cells[i].append(action)

            # 更新当前序列
            if self.current_time < self.num_hours:
                new_data = self.data_matrix[:, self.current_time]
                self.current_seq[i] = np.roll(self.current_seq[i], -1, axis=0)
                self.current_seq[i][-1] = new_data

        state = self._get_state()
        reward = self._calculate_reward()
        done = self._check_done()

        self.current_time += 1

        return state, reward, done, {}

    def _calculate_reward(self):
        rewards = []
        for i in range(self.num_agents):
            selected_data = self.data_matrix[self.selected_cells[i], :]
            inference_error = self._calculate_inference_error(selected_data)
            if inference_error <= self.error_bound:
                rewards.append(
                    1.0 - len(self.selected_cells[i]) / self.num_cells)
            else:
                rewards.append(-1.0)
        return rewards

    def _calculate_inference_error(self, selected_data):
        selected_data_tensor = tl.tensor(selected_data)
        weights, factors = parafac(selected_data_tensor, rank=min(
            selected_data.shape[0], selected_data.shape[1]), init='random')

        inferred_data = cp_to_tensor((weights, factors))
        true_data = np.mean(self.data_matrix, axis=0)
        error = np.abs(inferred_data - true_data).mean()
        return error

    def _check_done(self):
        # 检查是否满足质量要求或时间结束
        return any(len(cells) >= self.quality_threshold for cells in self.selected_cells) or self.current_time >= self.num_hours


# 定义纳什Q学习网络


class NashDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(NashDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 训练纳什Q学习模型
# 训练纳什Q学习模型


def train_nash_q(env, num_agents, num_episodes, gamma, epsilon, lr):
    state_size = env.observation_space.shape[0]

    # 获取每个动作的空间大小
    action_size = env.action_space.nvec[0]

    policy_nets = [NashDQN(state_size, action_size) for _ in range(num_agents)]
    optimizers = [optim.Adam(net.parameters(), lr=lr) for net in policy_nets]
    criterion = nn.MSELoss()

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            actions = []
            for i in range(num_agents):
                state_tensor = torch.tensor(
                    state, dtype=torch.float32).unsqueeze(0)
                if random.random() < epsilon:
                    action = random.choice(range(action_size))
                else:
                    with torch.no_grad():
                        q_values = policy_nets[i](state_tensor)
                        action = q_values.argmax().item()
                actions.append(action)

            next_state, reward, done, _ = env.step(actions)

            for i in range(num_agents):
                # 更新策略网络
                optimizer = optimizers[i]
                optimizer.zero_grad()

                state_tensor = torch.tensor(
                    state, dtype=torch.float32).unsqueeze(0)
                next_state_tensor = torch.tensor(
                    next_state, dtype=torch.float32).unsqueeze(0)

                q_values = policy_nets[i](state_tensor)
                next_q_values = policy_nets[i](
                    next_state_tensor).max(1)[0].detach()
                expected_q_values = reward + gamma * next_q_values

                loss = criterion(q_values, expected_q_values.unsqueeze(1))
                loss.backward()
                optimizer.step()

            state = next_state


# 初始化环境
num_agents = 36  # 每个cell是一个智能体
error_bound = 9 / 36
quality_threshold = 0.9 * data_matrix.shape[1]
env = MultiAgentCellSelectionEnv(
    data_matrix, error_bound, quality_threshold, lstm_model, seq_length, 36)

# 训练纳什Q学习模型
num_episodes = 100
gamma = 0.9
epsilon = 0.1
lr = 0.001
train_nash_q(env, num_agents, num_episodes, gamma, epsilon, lr)
