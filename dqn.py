import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    """简单的三层全连接神经网络，用于估计 Q 值"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存储一条经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """随机采样一个批次的经验"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN 智能体，包含策略网络和目标网络"""
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.99, hidden_dim: int = 128):
        """
        :param state_dim: 状态维度
        :param action_dim: 动作维度
        :param lr: 学习率
        :param gamma: 折扣因子
        :param hidden_dim: 隐藏层神经元数量
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.action_dim = action_dim

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        根据 epsilon-贪婪策略选择动作
        :param state: 当前状态
        :param epsilon: 探索率
        :return: 动作索引
        """
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def update(self, replay_buffer: ReplayBuffer, batch_size: int):
        """从经验回放中采样并更新策略网络"""
        if len(replay_buffer) < batch_size:
            return
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # 转换为张量
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # 计算当前 Q 值
        current_q = self.policy_net(state).gather(1, action)

        # 计算目标 Q 值
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (1 - done) * self.gamma * next_q

        # 损失函数（均方误差）
        loss = F.mse_loss(current_q, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        """将目标网络参数更新为策略网络参数"""
        self.target_net.load_state_dict(self.policy_net.state_dict())