# dqn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    """
    三层全连接神经网络，用于高精度的 Q 值估计
    """
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
    """
    经验回放缓冲区：增加存储动作掩码的字段
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, action_mask):
        """存储一条带有动作掩码的完整游戏经验"""
        self.buffer.append((state, action, reward, next_state, done, action_mask))

    def sample(self, batch_size: int):
        """随机采样一个批次的经验"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, action_mask = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, action_mask

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    升级版 Double DQN 智能体：深度集成动作掩码 (Action Masking) 机制
    """
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 建立 Policy Net (策略网络) 和 Target Net (目标网络)
        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, action_mask: np.ndarray, epsilon: float = 0.0) -> int:
        """
        核心机制：结合动作掩码选择动作。强制将不合法动作的胜率降为负无穷。
        """
        # 提取合法动作索引
        valid_indices = np.where(action_mask == 1)[0]
        
        # 容错：如果极极端情况下没有合法动作，被迫全盘解锁
        if len(valid_indices) == 0:
            valid_indices = np.arange(self.action_dim)

        # 探索探索 (Epsilon-Greedy)：只在合法动作池中随机挑选！绝不乱砸通货
        if np.random.rand() <= epsilon:
            return int(np.random.choice(valid_indices))

        # 利用神经网络进行贪婪选择
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t).cpu().numpy()[0]
        
        # 【降维打击的关键】：将所有非法的动作 Q 值设为极小的负数（-1e9）
        # 使得神经网络无论如何都不会去触发非法操作
        masked_q_values = np.where(action_mask == 1, q_values, -1e9)
        return int(masked_q_values.argmax())

    def update(self, replay_buffer: ReplayBuffer, batch_size: int):
        """
        基于 Double DQN 算法从经验回放中采样并更新网络参数
        """
        if len(replay_buffer) < batch_size:
            return
            
        state, action, reward, next_state, done, action_mask = replay_buffer.sample(batch_size)

        # 数据流转移至计算芯片 (GPU/CPU)
        state_t = torch.FloatTensor(state).to(self.device)
        action_t = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward_t = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        done_t = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        action_mask_t = torch.FloatTensor(action_mask).to(self.device)

        # 1. 计算当前网络评估的当前 Q 值
        current_q = self.policy_net(state_t).gather(1, action_t)

        # 2. 计算目标网络的目标 Q 值 (采用更加稳健的 Double DQN 算法)
        with torch.no_grad():
            # A) 用当前策略网络选出下一个状态表现最好的【合法动作】
            next_q_policy = self.policy_net(next_state_t)
            # 在评估 next 状态时同样注入掩码拦截
            masked_next_q_policy = torch.where(action_mask_t == 1, next_q_policy, torch.tensor(-1e9).to(self.device))
            best_actions = masked_next_q_policy.argmax(1).unsqueeze(1)
            
            # B) 用目标网络来计算这个动作的实际估值，有效防止过分乐观
            next_q_target = self.target_net(next_state_t).gather(1, best_actions)
            target_q = reward_t + (1 - done_t) * self.gamma * next_q_target

        # 3. 计算 Huber 损失（比 MSE 损失在面对脸黑随机波动时更具鲁棒性）
        loss = F.smooth_l1_loss(current_q, target_q)

        # 4. 反向传播与梯度削减 (防止梯度爆炸)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def update_target_network(self):
        """将评估网络的权重无缝对齐同步到目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())