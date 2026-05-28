# evaluate.py

import torch
import numpy as np
from env import GameEnv
from dqn import DQNAgent
from utils import GameData
import json

def generate_optimal_route(env: GameEnv, agent: DQNAgent, max_attempts: int = 20, max_steps: int = 50):
    """
    100% 确定性策略推演的最佳路线。
    因为游戏本身带随机性（如增幅、混沌是随机给词缀），我们让 AI 尝试最多 max_attempts 次做装，
    并只筛选出【最终真正完美通关、且消耗成本最低】的那一条纯净天命路线展示给玩家。
    """
    best_route = []
    min_cost = float('inf')

    for attempt in range(max_attempts):
        state = env.reset()
        current_route = []
        total_cost = 0
        step = 0
        done = False

        while not done and step < max_steps:
            action_mask = env.get_valid_actions()
            action_idx = agent.select_action(state, action_mask, epsilon=0.0)
            action = env.actions[action_idx]
            
            # 记录执行动作前的装备状态快照
            current_equipment_status = env.render()
            
            # 执行动作
            next_state, reward, done, info = env.step(action_idx)
            cost = int(action.price)
            
            step_info = {
                'step': step + 1,
                'action': action.name,
                'cost': cost,
                'equipment_status': current_equipment_status,
                'rarity': env.rarity,
                'affixes': [a.name for a in env.current_affixes] # 记录当前词缀组合
            }
            current_route.append(step_info)
            total_cost += cost
            
            state = next_state
            step += 1

        # 【逻辑清洗过滤门】：
        # 1. 必须是成功做出了神装的路线 (done == True)
        # 2. 我们追求成本更低的“欧皇路线”（排除掉中间被混沌石、重铸石洗坏了、反复折腾了 40 步的冗余路线）
        if done and total_cost < min_cost:
            min_cost = total_cost
            best_route = current_route

    # --- 智能化路线后处理：剔除无效的中间无效状态 ---
    # 如果路线上某个动作完全把装备重置了（比如用了重铸石），那么重铸石之前的动作对最终成品是没有贡献的废动作
    cleaned_route = []
    if best_route:
        # 逆向寻找最后一次大重置（比如 reroll_all 动作，或者稀有度归零）
        last_reset_idx = 0
        for i, step_data in enumerate(best_route):
            # 假设你的 items.json 中重铸石或混沌石的效果类型是 reroll_all
            # 如果使用了重铸/混沌这类洗全身的通货，它前面的步骤在“展示”上都可以舍弃
            if "重铸" in step_data['action'] or "混沌" in step_data['action']:
                last_reset_idx = i
        
        # 只保留最后一次洗底之后、一路变强的核心干净路线
        raw_cleaned = best_route[last_reset_idx:]
        
        # 重新编排步骤序号
        for new_step_idx, step_data in enumerate(raw_cleaned):
            step_data['step'] = new_step_idx + 1
            cleaned_route.append(step_data)
            
    return cleaned_route, min_cost if cleaned_route else 0


if __name__ == "__main__":
    # 保留原有的脚本测试入口，方便单独调试
    with open("config/training.json", "r", encoding="utf-8") as f:
        train_config = json.load(f)
    with open("config/equipment.json", "r", encoding="utf-8") as f:
        equip_config = json.load(f)

    game_data = GameData("config/items.json", "config/affixes.json")
    env = GameEnv(game_data, equip_config, train_config['reward'])

    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.num_actions)
    
    try:
        agent.policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=agent.device))
        agent.policy_net.eval()
        route, total_cost = generate_optimal_route(env, agent)
        print(f"过滤后的天命纯净做装成本: {total_cost}")
    except FileNotFoundError:
        print("请先训练模型。")