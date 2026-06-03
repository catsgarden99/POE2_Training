import sys, json
sys.path.insert(0, '.')
from env import GameEnv
from utils import GameData

gd = GameData('config/items_simple_currency.json', 'config/affixes/equipment_types/ring.json')
env = GameEnv(gd, {'target_prefixes': ['maximum_life'], 'target_suffixes': ['fire_resistance'],
                    'max_prefix': 3, 'max_suffix': 3}, {'success_bonus': 10000})

def fa(s):
    for a in env.actions:
        if s in a.name:
            return a.id
    return None

print('Total actions:', env.num_actions)

env.reset()
env.step(fa('蜕变'))  # normal -> magic (rarity 0->1)
print(f'After transmute: rarity={env.rarity} affixes={len(env.current_affixes)}')

env.step(fa('绝缘'))  # essence -> rare with fire res
print(f'After essence: rarity={env.rarity} affixes={len(env.current_affixes)}')
for a in env.current_affixes:
    print(f'  {a.name} pref={a.is_prefix}')

env.step(fa('崇高'))
print(f'After exalt: {len(env.current_affixes)}')
env.step(fa('浮夸'))
print(f'After hysteria: {len(env.current_affixes)}')
for a in env.current_affixes:
    print(f'  {a.name}')

with open('config/affixes/shared/essences.json', 'r', encoding='utf-8') as f:
    ess = json.load(f)
es = ess['essences']
print(f'\nessences.json: {len(es)} entries')
tiers = {}
for v in es.values():
    t = v.get('tier', '?')
    tiers[t] = tiers.get(t, 0) + 1
print(f'By tier: {tiers}')
print('OK')
