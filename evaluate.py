import torch
import numpy as np
from env import GameEnv
from dqn import DQNAgent
from utils import GameData
import json

def generate_optimal_route(env, agent, max_steps=50):
    state = env.reset()
    route = []
    total_cost = 0
    step = 0
    done = False
    illegal_count = 0  # 连续非法动作计数

    while not done and step < max_steps:
        action_idx = agent.select_action(state, epsilon=0.0)
        action = env.actions[action_idx]
        next_state, reward, done, info = env.step(action_idx)

        # 计算实际成本（扣除成功奖励）
        cost = -reward
        if done and 'success_bonus' in env.reward_config:
            cost -= env.reward_config['success_bonus']

        step_info = {
            'step': step + 1,
            'state': {
                'rarity': env.rarity,
                'prefixes': env.prefixes.copy(),
                'suffixes': env.suffixes.copy(),
                'target_prefix': env.target_prefix_count,
                'target_suffix': env.target_suffix_count,
                'need_prefix': env.target_prefix_goal,
                'need_suffix': env.target_suffix_goal
            },
            'action': action.name,
            'cost': cost,
            'valid': info['valid']
        }
        route.append(step_info)
        total_cost += cost
        state = next_state
        step += 1

        # 如果连续多次非法，可能是模型未学会，提前终止
        if not info['valid']:
            illegal_count += 1
            if illegal_count >= 5:
                print("检测到连续5次非法动作，模型可能未学会有效策略，提前终止")
                break
        else:
            illegal_count = 0

        if done:
            break

    return route, total_cost

if __name__ == "__main__":
    # 加载配置
    with open("config/training.json", "r", encoding="utf-8") as f:
        train_config = json.load(f)
    with open("config/equipment.json", "r", encoding="utf-8") as f:
        equip_config = json.load(f)

    game_data = GameData("config/items.json", "config/affixes.json")
    env = GameEnv(game_data, equip_config, train_config['reward'])

    state_dim = 9
    action_dim = env.num_actions
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=train_config['agent']['learning_rate'],
        gamma=train_config['agent']['gamma'],
        hidden_dim=train_config['agent']['hidden_dim']
    )
    agent.policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=agent.device))
    agent.policy_net.eval()

    route, total_cost = generate_optimal_route(env, agent)

    print("=" * 60)
    print(f"目标：获得 {env.target_prefix_goal} 条目标前缀和 {env.target_suffix_goal} 条目标后缀")
    print("=" * 60)
    for step in route:
        print(f"步骤 {step['step']}:")
        print(f"  动作：{step['action']}")
        print(f"  成本：{step['cost']:.2f}")
        print(f"  当前状态：稀有度 {step['state']['rarity']}, "
              f"前缀 {step['state']['prefixes']}, "
              f"后缀 {step['state']['suffixes']}")
        print(f"  已获得目标前缀/后缀：{step['state']['target_prefix']}/{step['state']['target_suffix']}")
        print()
    print(f"总成本：{total_cost:.2f}")
    print("=" * 60)