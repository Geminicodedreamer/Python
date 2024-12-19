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
import torch.nn.functional as F

# 定义自注意力机制


class SpatialAttention(nn.Module):
    def __init__(self, input_dim):
        super(SpatialAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.FloatTensor([input_dim]))

    def forward(self, x):
        Q = self.query(x)  # (batch_size, seq_length, num_cells)
        K = self.key(x)    # (batch_size, seq_length, num_cells)
        V = self.value(x)  # (batch_size, seq_length, num_cells)

        print(Q.shape)
        print(K.shape)
        print(V.shape)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)
        return attention_output

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
scaler = MinMaxScaler()
data_matrix = scaler.fit_transform(data_matrix.T).T

# 数据划分
train_data = data_matrix[:, :9*24]
test_data = data_matrix[:, 9*24:]

# 训练LSTM模型
seq_length = 24
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

# 定义DQN网络和经验回放池


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
        # selected_data = np.empty((264, 36))
        # selected_data.fill(np.nan)
        # for i in self.selected_cells:
        #     selected_data[i] = self.data_matrix[i]
        # # print(selected_data)
        selected_data = self.data_matrix[self.selected_cells, :]

        inference_error = self._calculate_inference_error(selected_data)
        if inference_error <= self.error_bound:
            return 1.0 - len(self.selected_cells) / self.num_cells
        else:
            return -1.0

    def _calculate_inference_error(self, selected_data):
        selected_data_tensor = tl.tensor(selected_data)
        weights, factors = parafac(selected_data_tensor, rank=min(
            selected_data.shape[0], selected_data.shape[1]), init='random')

        inferred_data = cp_to_tensor((weights, factors))

        true_data = np.mean(self.data_matrix, axis=0)   # 1 * 264matrix

        error = np.abs(inferred_data - true_data).mean()
        return error

    def _check_done(self):
        return len(self.selected_cells) >= self.quality_threshold or self.current_time >= self.num_hours

    def calculate_full_inference_and_error(self):
        selected_data = self.data_matrix[self.selected_cells, :]
        selected_data_tensor = tl.tensor(selected_data)
        weights, factors = parafac(selected_data_tensor, rank=min(
            selected_data.shape[0], selected_data.shape[1]), init='random')

        inferred_data = cp_to_tensor((weights, factors))
        mean_inferred_data = np.mean(inferred_data, axis=0)  # 形状为 (1, 48)
        true_data = np.mean(self.data_matrix, axis=0)

        inference_error = np.abs(mean_inferred_data - true_data).mean()

        num_cells_within_error = sum(
            np.abs(mean_inferred_data - true_data) <= self.error_bound)

        return mean_inferred_data, inference_error, num_cells_within_error


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


# 训练DQN模型
def train_dqn(env, num_episodes, gamma, epsilon, lr, memory_size=10000, batch_size=64, target_update=10, train=True):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(memory_size)
    episode_rewards = []

    for i_episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for t in range(env.num_hours):
            if random.random() < epsilon:
                action = random.randrange(action_size)
            else:
                with torch.no_grad():
                    action = policy_net(torch.tensor(
                        state, dtype=torch.float32)).argmax().item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            if train:
                memory.push(state, action, reward, next_state)
                state = next_state

                if len(memory) > batch_size:
                    experiences = memory.sample(batch_size)
                    batch_state, batch_action, batch_reward, batch_next_state = zip(
                        *experiences)

                    batch_state = torch.tensor(
                        batch_state, dtype=torch.float32)
                    batch_action = torch.tensor(
                        batch_action, dtype=torch.int64).unsqueeze(1)
                    batch_reward = torch.tensor(
                        batch_reward, dtype=torch.float32)
                    batch_next_state = torch.tensor(
                        batch_next_state, dtype=torch.float32)

                    current_q_values = policy_net(
                        batch_state).gather(1, batch_action)
                    max_next_q_values = target_net(
                        batch_next_state).max(1)[0].detach()
                    expected_q_values = batch_reward + gamma * max_next_q_values

                    loss = F.mse_loss(current_q_values,
                                      expected_q_values.unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if done:
                    break

        episode_rewards.append(total_reward)

        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f'Episode {i_episode},  Selected Cells Ratio: {
              len(env.selected_cells)}')
        if i_episode % 10 == 0:
            print(f"Episode {i_episode}, Reward: {total_reward}")

    return policy_net, episode_rewards


error_bound = 0.1
# quality_threshold = 48 * 0.9
quality_threshold = 216 * 0.9
env = CellSelectionEnvWithLSTM(
    test_data, error_bound, quality_threshold, lstm_model_with_attention, seq_length)
gamma = 0.99
epsilon = 0.1
lr = 0.001
num_episodes = 200

policy_net, episode_rewards = train_dqn(
    env, num_episodes, gamma, epsilon, lr, train=True)

plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

quality_threshold = 48 * 0.9
# 测试DQN模型
env_test = CellSelectionEnvWithLSTM(
    test_data, error_bound, quality_threshold, lstm_model_with_attention, seq_length, test=True)

state = env_test.reset()
total_reward = 0
done = False


while not done:
    with torch.no_grad():
        action = policy_net(torch.tensor(
            state, dtype=torch.float32)).argmax().item()
        # print(action)
    next_state, reward, done, _ = env_test.step(action)
    total_reward += reward
    state = next_state

    # 添加条件以避免无限循环或越界
    if env_test.current_time >= env_test.data_matrix.shape[1]:
        done = True

print(env_test.selected_cells)
# 结果展示
inferred_data, inference_error, num_cells_within_error = env_test.calculate_full_inference_and_error()
print(f"Inference Error: {inference_error}")
print(f"Number of Cells within Error Bound: {num_cells_within_error}")
plt.plot(inferred_data, label='Inferred Data')
plt.plot(np.mean(test_data, axis=0), label='True Data')
plt.legend()
plt.show()
