import pylab
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

# 定义LSTM模型


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# 创建序列数据的函数


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# 加载和预处理数据 (假设 data_matrix 是加载的数据矩阵)
data = pd.read_csv('../PM25.csv', header=None).values

# 将数据转换成36x264的矩阵
data_matrix = data.reshape((36, 264))

# 数据标准化
scaler = MinMaxScaler()

data_matrix = scaler.fit_transform(data_matrix.T).T

# 数据划分
train_data = data_matrix[:, :9*24]  # 前9天的数据用于训练
test_data = data_matrix[:, 9*24:]  # 后2天的数据用于测试

# 训练LSTM模型
seq_length = 5  # 设置序列长度
input_size = train_data.shape[0]
hidden_size = 64
num_layers = 2
output_size = train_data.shape[0]
num_epochs = 50

train_x, train_y = create_sequences(train_data.T, seq_length)
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)

lstm_model = LSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    lstm_model.train()
    optimizer.zero_grad()
    output = lstm_model(train_x)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 定义DQN相关的网络和经验回放池


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class CellSelectionEnvWithLSTM(gym.Env):
    def __init__(self, data_matrix, error_bound, quality_threshold, lstm_model, seq_length, test=False):
        super(CellSelectionEnvWithLSTM, self).__init__()
        self.data_matrix = data_matrix
        self.num_cells = data_matrix.shape[0]
        self.num_hours = data_matrix.shape[1]
        self.error_bound = error_bound
        self.quality_threshold = quality_threshold
        self.lstm_model = lstm_model
        self.seq_length = seq_length
        self.selected_cells = []
        self.current_time = 0
        self.test = test  # 标记是否是测试环境

        # 动作空间：选择一个小区
        self.action_space = spaces.Discrete(self.num_cells)

        # 状态空间：LSTM输出的状态表示
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_cells,), dtype=np.float32)

    def reset(self):
        self.selected_cells = []
        self.current_seq = np.zeros((self.seq_length, self.num_cells))
        self.current_time = 0
        return self._get_state()

    def _get_state(self):
        self.lstm_model.eval()
        with torch.no_grad():
            state = self.lstm_model(torch.tensor(
                self.current_seq, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        return state

    def step(self, action):
        if action not in self.selected_cells:
            self.selected_cells.append(action)

        # 更新当前序列
        new_data = self.data_matrix[:, self.current_time]
        self.current_seq = np.roll(self.current_seq, -1, axis=0)
        self.current_seq[-1] = new_data

        state = self._get_state()
        reward = self._calculate_reward()
        done = self._check_done()

        self.current_time += 1

        return state, reward, done, {}

    def _calculate_reward(self):
        # 根据选定的cell计算奖励
        selected_data = self.data_matrix[self.selected_cells, :]
        inference_error = self._calculate_inference_error(selected_data)
        if inference_error <= self.error_bound:
            return 1.0 - len(self.selected_cells) / self.num_cells
        else:
            return -1.0

    def _calculate_inference_error(self, selected_data):
        # 使用张量分解计算推断误差
        selected_data_tensor = tl.tensor(selected_data)
        weights, factors = parafac(selected_data_tensor, rank=min(
            selected_data.shape[0], selected_data.shape[1]), init='random')

        inferred_data = cp_to_tensor((weights, factors))
        true_data = np.mean(self.data_matrix, axis=0)
        error = np.abs(inferred_data - true_data).mean()
        return error

    def _check_done(self):
        # 检查是否满足质量要求或时间结束
        return len(self.selected_cells) >= self.quality_threshold or self.current_time >= self.num_hours


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 定义DQN训练和测试函数


def train_dqn(env, num_episodes, gamma, epsilon, lr, memory_size=10000, batch_size=64, target_update=10, train=True):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
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

            # 测试阶段不再进行探索
            if train:
                if random.random() < epsilon:
                    action = random.choice(range(action_size))
                else:
                    with torch.no_grad():
                        q_values = policy_net(state_tensor)
                        action = q_values.argmax().item()
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            if train:
                memory.push(state, action, reward, next_state)

                # 从经验回放中采样并训练网络
                if len(memory) >= batch_size:
                    transitions = memory.sample(batch_size)
                    batch_state, batch_action, batch_reward, batch_next_state = zip(
                        *transitions)

                    batch_state = torch.tensor(
                        batch_state, dtype=torch.float32)
                    batch_action = torch.tensor(
                        batch_action, dtype=torch.long).unsqueeze(1)
                    batch_reward = torch.tensor(
                        batch_reward, dtype=torch.float32).unsqueeze(1)
                    batch_next_state = torch.tensor(
                        batch_next_state, dtype=torch.float32)

                    current_q_values = policy_net(
                        batch_state).gather(1, batch_action)
                    next_q_values = target_net(batch_next_state).max(1)[
                        0].detach().unsqueeze(1)
                    expected_q_values = batch_reward + (gamma * next_q_values)

                    loss = criterion(current_q_values, expected_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # 更新目标网络
                if t % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            state = next_state
            episode_rewards.append(reward)

            if done:
                break

        rewards.append(sum(episode_rewards))
        selected_cells_ratios.append(len(env.selected_cells) / env.num_cells)

        print(f'Episode {episode}, Reward: {sum(episode_rewards)}, Selected Cells Ratio: {
              len(env.selected_cells) / env.num_cells}')

    return policy_net, rewards, selected_cells_ratios


# 训练DQN模型
gamma = 0.99
epsilon = 0.1
lr = 0.001
num_episodes = 300

# 使用前9天的数据进行训练
env_train = CellSelectionEnvWithLSTM(
    train_data, error_bound=0.1, quality_threshold=48*0.9, lstm_model=lstm_model, seq_length=seq_length)
policy_net, train_rewards, train_selected_cells_ratios = train_dqn(
    env_train, num_episodes, gamma, epsilon, lr)

# 使用后2天的数据进行测试，不再进行探索
env_test = CellSelectionEnvWithLSTM(
    test_data, error_bound=0.1, quality_threshold=212*0.9, lstm_model=lstm_model, seq_length=seq_length)
_, test_rewards, test_selected_cells_ratios = train_dqn(
    env_test, num_episodes, gamma, epsilon, lr, train=False)

# 绘制训练奖励曲线
plt.plot(train_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Rewards')
plt.show()

# 绘制选定单元格比例曲线
plt.plot(train_selected_cells_ratios[-48*100:])  # 只绘制后面的时间步
plt.xlabel('Time Step')
plt.ylabel('Selected Cells Ratio')
plt.title('Selected Cells Ratio During Training')
plt.show()

# 绘制测试奖励曲线
plt.plot(test_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Test Rewards')
plt.show()

# 绘制测试阶段选定单元格比例曲线
plt.plot(test_selected_cells_ratios[-212*100:])  # 只绘制后面的时间步
plt.xlabel('Time Step')
plt.ylabel('Selected Cells Ratio')
plt.title('Selected Cells Ratio During Testing')
plt.show()
