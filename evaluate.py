# evaluate.py

import torch
import numpy as np
from env import GameEnv
from dqn import DQNAgent
from utils import GameData
import json


def load_project_config(config_path: str = "config/project.json") -> dict:
    """加载项目主配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
            
            # 执行动作
            next_state, reward, done, info = env.step(action_idx)
            cost = int(action.price)
            
            # 关键修改：在执行动作后记录装备状态（展示操作结果）
            current_equipment_status = env.render()
            
            step_info = {
                'step': step + 1,
                'action': action.name,
                'cost': cost,
                'equipment_status': current_equipment_status,  # 操作后的状态
                'rarity': env.rarity,
                'affixes': [a.name for a in env.current_affixes]  # 操作后的词缀
            }
            current_route.append(step_info)
            total_cost += cost
            
            state = next_state
            step += 1

        # 【逻辑清洗过滤门】：
        if done and total_cost < min_cost:
            min_cost = total_cost
            best_route = current_route

    # --- 智能化路线后处理：剔除无效的中间无效状态 ---
    cleaned_route = []
    if best_route:
        last_reset_idx = 0
        for i, step_data in enumerate(best_route):
            if "重铸" in step_data['action'] or "混沌" in step_data['action']:
                last_reset_idx = i
        
        raw_cleaned = best_route[last_reset_idx:]
        
        for new_step_idx, step_data in enumerate(raw_cleaned):
            step_data['step'] = new_step_idx + 1
            cleaned_route.append(step_data)
            
    return cleaned_route, min_cost if cleaned_route else 0


if __name__ == "__main__":
    # 1. 加载项目主配置
    project_config = load_project_config()

    # 2. 根据主配置加载数据
    with open(project_config["training_params"], "r", encoding="utf-8") as f:
        train_config = json.load(f)
    with open(project_config["equipment_target"], "r", encoding="utf-8") as f:
        equip_config = json.load(f)

    game_data = GameData(
        project_config["currency_pool"],
        project_config["affixes_pool"]
    )
    env = GameEnv(game_data, equip_config, train_config['reward'])

    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.num_actions)
    
    try:
        agent.policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=agent.device))
        agent.policy_net.eval()
        route, total_cost = generate_optimal_route(env, agent)
        print(f"过滤后的天命纯净做装成本: {total_cost}")
    except FileNotFoundError:
        print("请先训练模型。")