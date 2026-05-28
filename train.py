# train.py

from env import GameEnv
from dqn import DQNAgent, ReplayBuffer
from utils import GameData
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def train(env: GameEnv, agent: DQNAgent, replay_buffer: ReplayBuffer, config: dict, writer: SummaryWriter):
    """
    带动作掩码的标准 Double DQN 训练主循环
    """
    num_episodes = config['training']['num_episodes']
    batch_size = config['training']['batch_size']
    target_update_freq = config['training']['target_update_freq']
    max_steps = config['training']['max_steps_per_episode']
    
    epsilon = config['epsilon']['initial']
    epsilon_min = config['epsilon']['min']
    epsilon_decay = config['epsilon']['decay_per_episode']

    episode_rewards = []
    success_count = 0

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        done = False

        while not done and step < max_steps:
            # 1. 核心改动：获取当前状态下的合法动作掩码
            action_mask = env.get_valid_actions()

            # 2. 将掩码传入智能体，保证探索和利用都绝对合法
            action_idx = agent.select_action(state, action_mask, epsilon)

            # 3. 环境步进
            next_state, reward, done, info = env.step(action_idx)

            # 4. 获取下一步状态的动作掩码，用于 Double DQN 计算目标 Q 值
            next_action_mask = env.get_valid_actions()

            # 5. 将带有掩码的经验存入缓冲区
            replay_buffer.push(state, action_idx, reward, next_state, done, next_action_mask)

            # 6. 智能体自我更新
            agent.update(replay_buffer, batch_size)

            state = next_state
            total_reward += reward
            step += 1

        # 统计成功率（根据环境 info 或最终状态判断）
        if done and total_reward > 0: 
            success_count += 1

        # 衰减探索率
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        # 定期同步目标网络
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # 记录 TensorBoard 日志
        writer.add_scalar('Reward/Train', total_reward, episode)
        writer.add_scalar('Epsilon', epsilon, episode)
        
        # 每 20 轮就打印一次，让进度条动得更频繁，并加入 flush=True 强行清空缓冲区吐给网页
        if (episode + 1) % 20 == 0:
            print(f"PROGRESS:Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.3f} | Last Reward: {total_reward:.1f}", flush=True)
    
    return episode_rewards


if __name__ == "__main__":
    # 加载静态游戏数据
    game_data = GameData("config/items.json", "config/affixes.json")

    # 加载全局训练配置与装备目标配置
    with open("config/training.json", "r", encoding="utf-8") as f:
        train_config = json.load(f)
    with open("config/equipment.json", "r", encoding="utf-8") as f:
        equip_config = json.load(f)

    # 初始化带动作掩码的游戏环境
    env = GameEnv(game_data, equip_config, train_config['reward'])

    state_dim = env.state_dim
    action_dim = env.num_actions

    # 创建智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=train_config['agent']['learning_rate'],\
        gamma=train_config['agent']['gamma'],
        hidden_dim=train_config['agent']['hidden_dim']
    )

    print(f"正在构建做装智能体。运行设备: {agent.device}")

    # 创建升级版经验回放区
    replay_buffer = ReplayBuffer(capacity=train_config['training']['replay_buffer_capacity'])
    writer = SummaryWriter('runs/crafting_experiment')

    # 启动训练
    print("开始强化学习重构训练...")
    rewards = train(env, agent, replay_buffer, train_config, writer)

    # 保存训练好的大脑模型
    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    print("训练完成！模型已成功固化为 dqn_model.pth")