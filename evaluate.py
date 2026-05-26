# evaluate.py

import torch
import numpy as np
from env import GameEnv
from dqn import DQNAgent
from utils import GameData
import json

def generate_optimal_route(env: GameEnv, agent: DQNAgent, max_steps: int = 50):
    """
    利用训练好的模型，100% 确定性地推演最佳期望做装路线
    """
    state = env.reset()
    route = []
    total_cost = 0
    step = 0
    done = False

    while not done and step < max_steps:
        # 核心改动：评估时同样必须获取合法动作掩码
        action_mask = env.get_valid_actions()
        
        # 强行令 epsilon=0.0，只选择神经网络认为期望最高的合法动作
        action_idx = agent.select_action(state, action_mask, epsilon=0.0)
        action = env.actions[action_idx]
        
        # 步进游戏
        next_state, reward, done, info = env.step(action_idx)

        # 规避惩罚项，计算纯粹的通货金钱消耗
        cost = action.price
        
        # 记录每一步的装备快照，供 GUI 可视化渲染
        step_info = {
            'step': step + 1,
            'action': action.name,
            'cost': cost,
            'valid': info.get('valid', True),
            'equipment_status': env.render() # 获取当前装备文字状态
        }
        route.append(step_info)
        total_cost += cost
        
        state = next_state
        step += 1

    return route, total_cost


if __name__ == "__main__":
    # 加载配置
    with open("config/training.json", "r", encoding="utf-8") as f:
        train_config = json.load(f)
    with open("config/equipment.json", "r", encoding="utf-8") as f:
        equip_config = json.load(f)

    game_data = GameData("config/items.json", "config/affixes.json")
    env = GameEnv(game_data, equip_config, train_config['reward'])

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.num_actions,
        lr=train_config['agent']['learning_rate'],
        gamma=train_config['agent']['gamma'],
        hidden_dim=train_config['agent']['hidden_dim']
    )
    
    # 加载训练好的权重模型
    try:
        agent.policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=agent.device))
        agent.policy_net.eval()
        print("🎉 成功成功加载本地做装大模型权重文件。")
        
        route, total_cost = generate_optimal_route(env, agent)

        print("=" * 60)
        print(f"📊 推演完成！达成目标预期总成本: {total_cost} 点通货价值")
        print("=" * 60)
        for s in route:
            print(f"【步骤 {s['step']}】: 使用了 [{s['action']}] (消耗成本: {s['cost']})")
            print(s['equipment_status'])
            print("-" * 40)
            
    except FileNotFoundError:
        print("未找到 dqn_model.pth 文件，请先运行 train.py 进行模型训练。")