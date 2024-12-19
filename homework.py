import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 设置随机种子以确保可复现性
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


class SparseMCS:
    def __init__(self, data, epsilon=0.25, p=0.3):
        self.data = data.values
        self.epsilon = epsilon
        self.p = p
        self.state_size = (2, data.shape[1])  # (2, number of time steps)
        self.action_size = data.shape[0]  # Number of cells
        self.reset()

    def reset(self):
        self.selected_cells = []
        self.state = np.zeros(self.state_size)
        return self.state

    def step(self, action):
        reward = -1  # Default negative reward for each action due to cost
        if action < self.action_size and action not in self.selected_cells:
            self.selected_cells.append(action)
            self.state[1, :] = self.state[0, :]
            self.state[0, :] = self.data[action, :]
        else:
            raise ValueError("Action out of bounds or already selected")

        if self.check_quality():
            reward = 10  # Positive reward if quality requirement is met
            done = True
        elif len(self.selected_cells) >= self.action_size:
            done = True
        else:
            done = False

        return self.state, reward, done

    def check_quality(self):
        inferred_data = self.infer_data()
        n_cycles = self.data.shape[1]  # 总的感知周期数
        count = 0

        # 遍历每个周期 k
        for k in range(n_cycles):
            # 计算每个周期 k 的绝对误差
            errors = np.abs(self.data[:, k] - inferred_data[:, k])
            # 计算平均误差
            mean_error = np.mean(errors)
            # 如果平均误差小于等于 epsilon，则计数器增加
            if mean_error <= self.epsilon:
                count += 1

        # 检查满足误差条件的周期数是否大于等于 n * p
        return count >= self.p * n_cycles

    def infer_data(self):
        if len(self.selected_cells) == 0:
            return np.zeros_like(self.data)
        selected_data = self.data[self.selected_cells, :]
        inferred_data = np.mean(selected_data, axis=0)
        return np.tile(inferred_data, (self.data.shape[0], 1))


class DQNLSTM(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNLSTM, self).__init__()
        self.lstm = nn.LSTM(state_size[1], 64, batch_first=True)
        self.fc = nn.Linear(64, action_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), 64).to(x.device)  # Hidden state
        c_0 = torch.zeros(1, x.size(0), 64).to(x.device)  # Cell state
        x, _ = self.lstm(x, (h_0, c_0))
        x = self.fc(x[:, -1, :])
        return x


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNLSTM(state_size, action_size).to(self.device)
        self.target_model = DQNLSTM(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            target = reward
            if not done:
                target = reward + self.gamma * \
                    torch.max(self.target_model(next_state)[0]).item()
            target_f = self.model(state)
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)
        states = torch.cat(states)
        targets = torch.cat(targets)
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(states), targets)
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# 加载数据集
data = pd.read_csv('../PM25.csv', header=None)
print(data.head())

# 初始化环境和智能体
env = SparseMCS(data)
agent = Agent(env.state_size, env.action_size)

# 训练参数
episodes = 10
batch_size = 32
update_target_frequency = 5

for e in range(episodes):
    total_reward = 0
    selection_sequence = []
    cells_selected_per_hour = []  # 存储每个小时选择的单元格数量
    for time in range(48, data.shape[1]):  # 从第49个小时开始循环
        state = env.reset()
        cells_selected_this_hour = 0
        while True:
            state = np.reshape(state, [1, *env.state_size])
            action = agent.act(state)
            try:
                next_state, reward, done = env.step(action)
            except ValueError:
                continue  # Skip this action if it's already selected or out of bounds
            next_state = np.reshape(next_state, [1, *env.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            cells_selected_this_hour += 1
            selection_sequence.append((time, action))
            error = np.mean(
                np.abs(env.data[:, :time+1] - env.infer_data()[:, :time+1]))
            # print(
            #     f"时间：{time}，选择的单元格：{action}，奖励：{reward}，是否完成：{done}，推断误差：{error}")
            if done or len(env.selected_cells) >= env.action_size:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if time % update_target_frequency == 0:
                agent.update_target_model()
        cells_selected_per_hour.append(cells_selected_this_hour)
        print(
            f"时间：{time}， 选择的单元格：{env.selected_cells}，占比：{len(env.selected_cells)} / {env.action_size}")
    avg_cells_selected_per_hour = np.average(cells_selected_per_hour)
    print(
        f"episode: {e}/{episodes}, total_reward: {total_reward}, e: {agent.epsilon:.2f}")
    print(f"Selected cells for this episodes: {selection_sequence}")
    print(
        f"Average cells selected per hour: {avg_cells_selected_per_hour:.2f}")
