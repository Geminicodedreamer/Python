import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from gym import spaces
import gym
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor

# 定义DQN模型


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

# 定义自注意力机制


class SpatialAttention(nn.Module):
    def __init__(self, input_dim):
        super(SpatialAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.FloatTensor([input_dim]))

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)
        return attention_output

# 定义经验回放


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

# 定义带注意力机制的LSTM模型


class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMWithAttention, self).__init__()
        self.attention = SpatialAttention(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.attention(x)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
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


# 加载和预处理数据
data = pd.read_csv('../PM25.csv', header=None).values
data_matrix = data.reshape((36, 264))

# 数据划分
train_data = data_matrix[:, :9*24]
test_data = data_matrix[:, 9*24:]

# 训练LSTM模型
seq_length = 5
input_size = train_data.shape[0]
hidden_size = 64
num_layers = 2
output_size = train_data.shape[0]
num_epochs = 50

train_x, train_y = create_sequences(train_data.T, seq_length)
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)

lstm_model_with_attention = LSTMWithAttention(
    input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model_with_attention.parameters(), lr=0.001)

for epoch in range(num_epochs):
    lstm_model_with_attention.train()
    optimizer.zero_grad()
    output = lstm_model_with_attention(train_x)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# L1最小化函数
    def l1_minimization(selected_data):
        num_cells, num_time_steps = selected_data.shape
        lasso = Lasso(alpha=0.1, max_iter=10000)

        # 创建随机测量矩阵A和观测值y
        A = np.random.randn(num_time_steps, num_cells)  # 随机测量矩阵

        # 生成稀疏信号并处理NaN值
        y = selected_data.sum(axis=0)  # 对每一列求和，形成稀疏信号

        # 使用SimpleImputer填充NaN值
        imputer = SimpleImputer(strategy='mean')
        y = imputer.fit_transform(y.reshape(-1, 1)).ravel()  # 将NaN填充为均值

        # 检查y是否为空或仅包含NaN
        if len(y) == 0 or np.isnan(y).all():
            print("Warning: No valid samples found after NaN handling.")
            return np.zeros(num_cells)  # 返回零向量作为默认值

        # 拟合LASSO模型
        try:
            lasso.fit(A, y)
        except ValueError as e:
            print(f"Error fitting LASSO: {e}")
            return np.zeros(num_cells)  # 返回零向量作为默认值

        # 使用系数作为重构的稀疏数据
        reconstructed_data = lasso.coef_
        return reconstructed_data


# 定义Cell Selection环境


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
        self.test = test

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

        self.current_time += 1

        # 添加检查，确保 current_time 不会超出 data_matrix 的时间步范围
        if self.current_time >= self.data_matrix.shape[1]:
            done = True
            reward = 0  # 或者根据需求设置合适的reward
            return self._get_state(), reward, done, {}

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
        selected_data = np.empty(self.data_matrix.shape)
        selected_data.fill(np.nan)
        for i in self.selected_cells:
            selected_data[i, :] = self.data_matrix[i, :]

        inference_error = self._calculate_inference_error(selected_data)
        if inference_error <= self.error_bound:
            return 1.0 - len(self.selected_cells) / self.num_cells
        else:
            return -1.0

    def _calculate_inference_error(self, selected_data):
        selected_data_tensor = tl.tensor(selected_data)
        rank = min(selected_data.shape[0], selected_data.shape[1])
        weights, factors = parafac(
            selected_data_tensor, rank=rank, init='random')

        inferred_data = cp_to_tensor((weights, factors))

        # 对 inferred_data 和 true_data 进行调整以匹配形状
        inferred_data_mean = np.nanmean(
            inferred_data, axis=0)  # 取平均值，形状为 (36,)
        true_data = np.nanmean(self.data_matrix, axis=0)       # 真实数据，形状为 (48,)

        # 对齐数据的长度
        min_length = min(len(inferred_data_mean), len(true_data))
        inferred_data_mean = inferred_data_mean[:min_length]
        true_data = true_data[:min_length]

        error = np.abs(inferred_data_mean - true_data).mean()
        return error

    def _check_done(self):
        return len(self.selected_cells) >= self.quality_threshold or self.current_time >= self.num_hours

    def handle_nan_values(data):
        """
        使用简单插值或均值填充来处理 NaN 值。
        """
        imputer = SimpleImputer(strategy='mean')
        return imputer.fit_transform(data)

    def calculate_full_inference_and_error(selected_data, data_matrix, error_bound):
        """
        完整推断和误差计算，使用张量分解并处理 SVD 不收敛的问题。
        """
        # 处理 NaN 值
        selected_data = handle_nan_values(selected_data)

        selected_data_tensor = tl.tensor(selected_data)

        # 降低张量分解的秩
        rank = min(selected_data.shape[0],
                   selected_data.shape[1], 10)  # 设定最大秩为 10

        try:
            weights, factors = parafac(
                selected_data_tensor, rank=rank, init='random')
        except np.linalg.LinAlgError as e:
            print(f"SVD did not converge: {e}")
            return np.nan, np.nan, 0  # 返回默认值，避免程序崩溃

        inferred_data = cp_to_tensor((weights, factors))
        mean_inferred_data = np.nanmean(inferred_data, axis=0)  # 形状为 (48,)
        # 真实数据，形状为 (48,)
        true_data = np.nanmean(data_matrix, axis=0)

        # 对齐 inferred_data 和 true_data 的长度
        min_length = min(len(mean_inferred_data), len(true_data))
        mean_inferred_data = mean_inferred_data[:min_length]
        true_data = true_data[:min_length]

        inference_error = np.abs(mean_inferred_data - true_data).mean()

        num_cells_within_error = sum(
            np.abs(mean_inferred_data - true_data) <= error_bound)

        return mean_inferred_data, inference_error, num_cells_within_error


# 训练DQN智能体


def train_dqn(env, num_episodes=100):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = ReplayMemory(10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, action_size)
            else:
                with torch.no_grad():
                    action = torch.argmax(policy_net(
                        torch.tensor(state, dtype=torch.float32))).item()

            next_state, reward, done, _ = env.step(action)
            memory.push(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state = zip(
                    *transitions)

                batch_state = torch.tensor(batch_state, dtype=torch.float32)
                batch_action = torch.tensor(
                    batch_action, dtype=torch.long).unsqueeze(1)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
                batch_next_state = torch.tensor(
                    batch_next_state, dtype=torch.float32)

                current_q_values = policy_net(
                    batch_state).gather(1, batch_action).squeeze()
                next_q_values = target_net(batch_next_state).max(1)[0].detach()
                target_q_values = batch_reward + (gamma * next_q_values)

                loss = F.mse_loss(current_q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f'Episode {episode}, Total Reward: {
                  total_reward}, Epsilon: {epsilon}')

    return policy_net


# 创建环境并训练
error_bound = 0.1
quality_threshold = 5
env = CellSelectionEnvWithLSTM(
    test_data, error_bound, quality_threshold, lstm_model_with_attention, seq_length)
trained_policy = train_dqn(env)

# 测试训练的智能体
state = env.reset()
done = False
while not done:
    action = torch.argmax(trained_policy(
        torch.tensor(state, dtype=torch.float32))).item()
    next_state, reward, done, _ = env.step(action)
    state = next_state

# 打印选中的小区
print(f"Selected cells: {env.selected_cells}")
# print(f"Final inference error: {env.calculate_full_inference_and_error()}")
