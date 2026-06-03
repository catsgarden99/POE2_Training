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

print('Actions:', env.num_actions)  # should be 17
print('State dim:', env.state_dim)  # should be 15

# Test: normal item -> can use essence? no
mask = env.get_valid_actions()
use_idx = fa('使用精华')
print(f'Use essence on normal: mask={mask[use_idx]}')  # should be 0

# Transmute -> magic
env.step(fa('蜕变'))
mask = env.get_valid_actions()
print(f'Use essence on magic: mask={mask[use_idx]}')  # should be 1 (life or fire res target)

# Use essence
s, r, d, info = env.step(use_idx)
print(f'After essence: rarity={env.rarity} affixes={len(env.current_affixes)} reward={r}')
for a in env.current_affixes:
    print(f'  {a.name} pref={a.is_prefix}')
print(f'State[13] (essence price): {s[13]}')  # should be > 0

# Fill item to rare 3 affixes
env.step(fa('崇高'))
env.step(fa('崇高'))
print(f'Full item: {len(env.current_affixes)} affixes')
mask = env.get_valid_actions()
print(f'Use essence on full item: mask={mask[use_idx]}')  # might be 0 if no free slot

# Validate essence data integrity
with open('config/affixes/shared/essences.json', 'r', encoding='utf-8') as f:
    ess = json.load(f)
es = ess['essences']
print(f'\nessences.json: {len(es)} entries')
print('OK')
