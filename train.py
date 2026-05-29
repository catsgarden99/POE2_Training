# train.py

from env import GameEnv
from dqn import DQNAgent, ReplayBuffer
from utils import GameData
import json
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt


def load_project_config(config_path: str = "config/project.json") -> dict:
    """加载项目主配置文件，获取各模块的配置路径"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
            action_mask = env.get_valid_actions()
            action_idx = agent.select_action(state, action_mask, epsilon)
            next_state, reward, done, info = env.step(action_idx)
            next_action_mask = env.get_valid_actions()
            replay_buffer.push(state, action_idx, reward, next_state, done, next_action_mask)
            agent.update(replay_buffer, batch_size)

            state = next_state
            total_reward += reward
            step += 1

        if done and total_reward > 0: 
            success_count += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if episode % target_update_freq == 0:
            agent.update_target_network()

        writer.add_scalar('Reward/Train', total_reward, episode)
        writer.add_scalar('Epsilon', epsilon, episode)
        
        if (episode + 1) % 20 == 0:
            print(f"PROGRESS:Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.3f} | Last Reward: {total_reward:.1f}", flush=True)
    
    return episode_rewards


if __name__ == "__main__":
    # 1. 加载项目主配置
    project_config = load_project_config()
    print(f"[CONFIG] 项目配置已加载: {project_config.get('description', '无描述')}")

    # 2. 根据主配置加载游戏数据
    game_data = GameData(
        project_config["currency_pool"],
        project_config["affixes_pool"]
    )

    # 3. 加载训练配置与装备目标
    with open(project_config["training_params"], "r", encoding="utf-8") as f:
        train_config = json.load(f)
    with open(project_config["equipment_target"], "r", encoding="utf-8") as f:
        equip_config = json.load(f)

    # 初始化带动作掩码的游戏环境
    env = GameEnv(game_data, equip_config, train_config['reward'])

    state_dim = env.state_dim
    action_dim = env.num_actions

    # 创建智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=train_config['agent']['learning_rate'],
        gamma=train_config['agent']['gamma'],
        hidden_dim=train_config['agent']['hidden_dim']
    )

    print(f"正在构建做装智能体。运行设备: {agent.device}")
    print(f"[INFO] 通货池: {project_config['currency_pool']} | 动作数量: {action_dim}")

    # 创建升级版经验回放区
    replay_buffer = ReplayBuffer(capacity=train_config['training']['replay_buffer_capacity'])
    writer = SummaryWriter('runs/crafting_experiment')

    # 启动训练
    print("开始强化学习重构训练...")
    rewards = train(env, agent, replay_buffer, train_config, writer)

    # 保存训练好的大脑模型
    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    print("训练完成！模型已成功固化为 dqn_model.pth")