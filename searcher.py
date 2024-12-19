import torch
import torch.optim as optim
import torch.nn as nn
import random
from gym import spaces
import gym
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 读取数据
data = pd.read_csv('../PM25.csv', header=None).values

# 将数据转换成36x264的矩阵
data_matrix = data.reshape((36, 264))

# 数据标准化
scaler = MinMaxScaler()
data_matrix = scaler.fit_transform(data_matrix.T).T

# 前2*24小时作为训练集，剩下的时间作为测试集
train_hours = 2 * 24
train_data = data_matrix[:, :train_hours]
test_data = data_matrix[:, train_hours:]


# 初始化环境参数
num_cells = data_matrix.shape[0]
error_bound = 9 / 36
quality_threshold = int(0.9 * num_cells)


class CellSelectionEnv(gym.Env):
    def __init__(self, data_matrix, error_bound, quality_threshold):
        super(CellSelectionEnv, self).__init__()
        self.data_matrix = data_matrix
        self.num_cells = data_matrix.shape[0]
        self.num_hours = data_matrix.shape[1]
        self.error_bound = error_bound
        self.quality_threshold = quality_threshold
        self.selected_cells = []

        # 动作空间：选择一个小区
        self.action_space = spaces.Discrete(self.num_cells)

        # 状态空间：已选择的小区
        self.observation_space = spaces.MultiBinary(self.num_cells)

    def reset(self):
        self.selected_cells = []
        return self._get_state()

    def _get_state(self):
        state = np.zeros(self.num_cells)
        state[self.selected_cells] = 1
        return state

    def step(self, action):
        if action not in self.selected_cells:
            self.selected_cells.append(action)
            # print("Selected cells:", self.selected_cells)

        state = self._get_state()
        reward = self._calculate_reward()
        done = self._check_done()

        return state, reward, done, {}

    def _calculate_reward(self):
        # 根据选定的小区计算奖励
        selected_data = self.data_matrix[self.selected_cells, :]
        inference_error = self._calculate_inference_error(selected_data)
        if inference_error <= self.error_bound:
            return 1.0 - len(self.selected_cells) / self.num_cells
        else:
            return -1.0

    def _calculate_inference_error(self, selected_data):
        # 使用简单平均法计算推断误差
        inferred_data = np.mean(selected_data, axis=0)
        true_data = np.mean(self.data_matrix, axis=0)
        error = np.abs(inferred_data - true_data).mean()
        return error

    def _check_done(self):
        # 检查是否满足质量要求
        return len(self.selected_cells) >= self.quality_threshold


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


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 24  # 使用过去24小时的数据预测下一个时间点
train_x, train_y = create_sequences(train_data.T, seq_length)
test_x, test_y = create_sequences(test_data.T, seq_length)

train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)


# 定义并训练LSTM模型
input_size = train_data.shape[0]
hidden_size = 64
num_layers = 2
output_size = train_data.shape[0]

lstm_model = LSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
num_epochs = 50

for epoch in range(num_epochs):
    lstm_model.train()
    optimizer.zero_grad()
    output = lstm_model(train_x)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_dqn(env, num_episodes, gamma, epsilon, lr):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        for t in range(env.num_hours):
            if random.random() < epsilon:
                action = random.choice(range(action_size))
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(
                next_state, dtype=torch.float32).unsqueeze(0)
            reward = torch.tensor([reward], dtype=torch.float32)

            if done:
                next_q_values = torch.zeros(1)
            else:
                with torch.no_grad():
                    next_q_values = target_net(next_state).max(1)[
                        0].unsqueeze(0)

            q_values = policy_net(state)
            q_value = q_values[0, action]
            target = reward + (gamma * next_q_values)

            loss = criterion(q_value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done:
                break

            state = next_state

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f'Episode {episode}, Loss: {loss.item()}')

        print(
            f'Episode {episode}, Selected Cells Ratio: {len(env.selected_cells)} / {env.num_cells}')
    return policy_net


# 初始化环境
env = CellSelectionEnv(data_matrix, error_bound, quality_threshold)

# 训练DQN模型
num_episodes = 100
gamma = 0.99
epsilon = 0.1
lr = 0.001
policy_net = train_dqn(env, num_episodes, gamma, epsilon, lr)


class CellSelectionEnvWithLSTM(gym.Env):
    def __init__(self, data_matrix, error_bound, quality_threshold, lstm_model, seq_length):
        super(CellSelectionEnvWithLSTM, self).__init__()
        self.data_matrix = data_matrix
        self.num_cells = data_matrix.shape[0]
        self.num_hours = data_matrix.shape[1]
        self.error_bound = error_bound
        self.quality_threshold = quality_threshold
        self.lstm_model = lstm_model
        self.seq_length = seq_length
        self.selected_cells = []

        # 动作空间：选择一个小区
        self.action_space = spaces.Discrete(self.num_cells)

        # 状态空间：LSTM输出的状态表示
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_cells,), dtype=np.float32)

    def reset(self):
        self.selected_cells = []
        self.current_seq = np.zeros((self.seq_length, self.num_cells))
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

        state = self._get_state()
        reward = self._calculate_reward()
        done = self._check_done()

        # 更新当前序列
        new_data = self.data_matrix[:, action]
        self.current_seq = np.roll(self.current_seq, -1, axis=0)
        self.current_seq[-1] = new_data

        return state, reward, done, {}

    def _calculate_reward(self):
        # 根据选定的小区计算奖励
        selected_data = self.data_matrix[self.selected_cells, :]
        inference_error = self._calculate_inference_error(selected_data)
        if inference_error <= self.error_bound:
            return 1.0 - len(self.selected_cells) / self.num_cells
        else:
            return -1.0

    def _calculate_inference_error(self, selected_data):
        # 使用简单平均法计算推断误差
        inferred_data = np.mean(selected_data, axis=0)
        true_data = np.mean(self.data_matrix, axis=0)
        error = np.abs(inferred_data - true_data).mean()
        return error

    def _check_done(self):
        # 检查是否满足质量要求
        return len(self.selected_cells) >= self.quality_threshold


# 初始化环境
env = CellSelectionEnvWithLSTM(
    data_matrix, error_bound, quality_threshold, lstm_model, seq_length)

# 训练DQN模型
policy_net = train_dqn(env, num_episodes, gamma, epsilon, lr)


def test_model(env, policy_net, num_episodes):
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0

        for t in range(env.num_hours):
            with torch.no_grad():
                q_values = policy_net(state)
                action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

    avg_reward = np.mean(rewards)
    print(f'Average Reward over {num_episodes} episodes: {avg_reward}')
    return rewards


# 测试模型
num_test_episodes = 100
rewards = test_model(env, policy_net, num_test_episodes)