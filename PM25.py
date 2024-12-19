import torch
import torch.optim as optim
import torch.nn as nn
import gym
import numpy as np
from gym import spaces  # 修正: 导入spaces
from sklearn.preprocessing import MinMaxScaler
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor
import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
data = pd.read_csv('../PM25.csv', header=None).values
data_matrix = data.reshape((36, 264))
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


seq_length = 48
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

# PPO部分


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # 策略网络输出每个动作的概率
        self.policy_head = nn.Linear(128, action_size)
        # 价值网络输出状态的价值
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

    def policy(self, x):
        x = self.forward(x)
        return torch.softmax(self.policy_head(x), dim=-1)

    def value(self, x):
        x = self.forward(x)
        return self.value_head(x)

# 定义环境


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
        self.current_time = 0

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
        true_data = np.mean(self.data_matrix, axis=0)
        error = np.abs(inferred_data - true_data).mean()
        return error

    def _check_done(self):
        return len(self.selected_cells) >= self.quality_threshold or self.current_time >= self.num_hours

# PPO算法实现


class PPOAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, clip_ratio=0.2, critic_loss_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.actor_critic = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.actor_critic.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def compute_returns(self, rewards, values, dones, next_value):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return returns

    def update(self, states, actions, log_probs, returns, values, entropy):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        returns = torch.tensor(returns)
        old_log_probs = torch.tensor(log_probs)
        advantages = returns - torch.tensor(values)

        # 获取当前策略的动作概率对数
        new_probs = self.actor_critic.policy(states)
        dist = torch.distributions.Categorical(new_probs)
        new_log_probs = dist.log_prob(actions)

        # 计算策略损失
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio,
                            1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 计算价值损失
        value_loss = self.critic_loss_coef * \
            (returns - self.actor_critic.value(states)).pow(2).mean()

        # 计算熵损失
        entropy = torch.stack(entropy)  # 将列表转换为张量
        entropy_loss = -self.entropy_coef * entropy.mean()  # 计算平均值

        # 总损失
        loss = policy_loss + value_loss + entropy_loss

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes, batch_size=64):
        rewards_all = []
        for episode in range(num_episodes):
            state = env.reset()
            states, actions, log_probs, rewards, values, entropies, dones = [], [], [], [], [], [], []

            for t in range(env.num_hours):
                action, log_prob, entropy = self.select_action(state)
                value = self.actor_critic.value(
                    torch.tensor(state, dtype=torch.float32))

                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                entropies.append(entropy)
                dones.append(done)

                state = next_state

                if done:
                    break

            # 获取下一状态的价值并计算回报
            next_value = self.actor_critic.value(
                torch.tensor(state, dtype=torch.float32)).item()
            returns = self.compute_returns(rewards, values, dones, next_value)

            # 批量更新策略和价值网络
            self.update(states, actions, log_probs, returns, values, entropies)

            episode_reward = sum(rewards)
            rewards_all.append(episode_reward)

            # 打印已选择单元格的比例和总数量
            print(f'Episode {episode}, Selected Cells Ratio: {
                  len(env.selected_cells)} / {env.num_cells}')

            if episode % 10 == 0:
                print(f'Episode {episode}, Total Reward: {episode_reward}')

        return rewards_all


# 初始化环境
state_size = data_matrix.shape[0]
action_size = data_matrix.shape[0]

error_bound = 9 / 36
quality_threshold = 0.9 * data_matrix.shape[1]
env = CellSelectionEnvWithLSTM(
    data_matrix, error_bound, quality_threshold, lstm_model, seq_length)

# 训练PPO模型
ppo_agent = PPOAgent(state_size, action_size)
train_rewards = ppo_agent.train(env, num_episodes=100)

# 绘制训练奖励曲线
plt.plot(train_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Rewards')
plt.show()
