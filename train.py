from env import GameEnv
from dqn import DQNAgent, ReplayBuffer
from utils import GameData
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def train(env, agent, replay_buffer, config, writer):
    """
    主训练循环
    :param env: 游戏环境
    :param agent: DQN 智能体
    :param replay_buffer: 经验回放缓冲区
    :param config: 训练配置字典
    :param writer: TensorBoard 写入器
    :return: 每个 episode 的总奖励列表
    """
    num_episodes = config['training']['num_episodes']
    batch_size = config['training']['batch_size']
    target_update_freq = config['training']['target_update_freq']
    max_steps = config['training']['max_steps_per_episode']
    epsilon = config['epsilon']['initial']
    epsilon_min = config['epsilon']['min']
    epsilon_decay = config['epsilon']['decay_per_episode']

    episode_rewards = []
    successes = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        while True:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.update(replay_buffer, batch_size)
            step += 1
            if done or step > max_steps:
                break

        episode_rewards.append(total_reward)
        writer.add_scalar('Reward/Total', total_reward, episode)
        writer.add_scalar('Epsilon', epsilon, episode)

        # 每个 episode 结束后衰减 epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 定期更新目标网络
        if episode % target_update_freq == 0:
            agent.update_target()

        # 记录成功次数
        if done and step <= max_steps:
            successes.append(1)
        else:
            successes.append(0)

        # 打印进度
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            success_rate = np.mean(successes[-100:])
            print(f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.3f}, Success rate (last 100): {success_rate:.2f}")

    writer.close()
    return episode_rewards

if __name__ == "__main__":
    # 加载训练配置
    with open("config/training.json", "r", encoding="utf-8") as f:
        train_config = json.load(f)

    # 加载游戏数据（道具和词缀）
    game_data = GameData("config/items.json", "config/affixes.json")

    # 加载装备配置
    with open("config/equipment.json", "r", encoding="utf-8") as f:
        equip_config = json.load(f)

    # 创建环境，传入奖励配置
    env = GameEnv(game_data, equip_config, train_config['reward'])

    # 状态维度（与 env._get_state 输出的维度一致）
    state_dim = 9
    action_dim = env.num_actions

    # 创建智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=train_config['agent']['learning_rate'],
        gamma=train_config['agent']['gamma'],
        hidden_dim=train_config['agent']['hidden_dim']
    )

    # 检查gpu    
    print(f"Device: {agent.device}")

    # 创建经验回放缓冲区
    replay_buffer = ReplayBuffer(capacity=train_config['training']['replay_buffer_capacity'])

    # TensorBoard 日志
    writer = SummaryWriter('runs/crafting_experiment')

    # 开始训练
    rewards = train(env, agent, replay_buffer, train_config, writer)

    # 保存模型
    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    print("模型已保存为 dqn_model.pth")

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label='Raw Reward')
    if len(rewards) >= 100:
        smoothed = np.convolve(rewards, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(rewards)), smoothed, 'r', label='100-episode avg')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()