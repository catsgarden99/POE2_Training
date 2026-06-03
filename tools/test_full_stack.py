import sys, json, numpy as np
sys.path.insert(0, '.')
from env import GameEnv
from utils import GameData

gd = GameData('config/items_simple_currency.json', 'config/affixes/equipment_types/ring.json')
env = GameEnv(gd, {'target_prefixes': ['maximum_life'], 'target_suffixes': ['fire_resistance'],
                    'max_prefix': 3, 'max_suffix': 3, 'equipment_type': 'ring'},
               {'success_bonus': 10000, 'failure_penalty': -1000, 'reward_scale': 100, 'max_budget': 5000})

def fa(s):
    for a in env.actions:
        if s in a.name: return a.id
    return None

print(f'=== State: {env.state_dim}dim | Actions: {env.num_actions} | Budget: {env.max_budget} ===')

state = env.reset()
print(f'[INIT] state dims: {len(state)}, budget: {env.budget_remaining}')
print(f'  [15] budget={state[15]:.3f}, [16] pre_quality={state[16]:.3f}, [17] suf_quality={state[17]:.3f}')
print(f'  [18] best_tier={state[18]:.3f}, [19] worst_tier={state[19]:.3f}, [20] avg_nt={state[20]:.3f}')
print(f'  [21] steps={state[21]:.3f}, [22] is_ring={state[22]:.0f}, [23] non_target={state[23]:.0f}')

# Transmute -> magic -> use essence
env.step(fa('蜕变'))
s, r, d, info = env.step(fa('使用精华'))
print(f'\n[ESSENCE] reward={r:.1f}, done={d}')
print(f'  after: rarity={env.rarity}, affixes={len(env.current_affixes)}')
for a in env.current_affixes:
    print(f'  {a.name} T{a.tier} pref={a.is_prefix}')
print(f'  state[16] pre_quality={s[16]:.3f}, [17] suf_quality={s[17]:.3f}')
print(f'  state[18] best_tier={s[18]:.3f}')

# Exalt x2 -> 3 prefixes 1 suffix
env.step(fa('崇高'))
env.step(fa('崇高'))
s, r, d, info = env.step(fa('崇高'))
print(f'\n[EXALTx3] reward={r:.1f}, affixes={len(env.current_affixes)}')
for a in env.current_affixes:
    print(f'  {a.name} T{a.tier}')
print(f'  state[16] pre_quality={s[16]:.3f}, [23] non_target={s[23]:.0f}')
print(f'  value={env._state_value():.3f}')
print(f'  budget_remaining={env.budget_remaining:.0f}')

# Validate train.py loads correctly
from train import load_project_config
pc = load_project_config()
print(f'\n[TRAIN] project config points to: {pc["affixes_pool"]}')
gd2 = GameData(pc["currency_pool"], pc["affixes_pool"])
print(f'  loaded actions: {len(gd2.actions)}')

from dqn import DQNAgent, ReplayBuffer
state_dim = env.state_dim
action_dim = env.num_actions
agent = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, hidden_dim=128)
print(f'  DQN: state={state_dim} action={action_dim} hidden=128')
print('\nALL OK')
