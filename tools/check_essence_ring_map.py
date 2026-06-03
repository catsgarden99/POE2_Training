print("=== 精华 vs ring.json 对照表 ===")
import json
ring = json.load(open('config/affixes/equipment_types/ring.json','r',encoding='utf-8'))

# 精华 -> 精华给的数值范围 -> ring.json的对应词缀+对应T阶
checks = {
  "大火焰精华": {"poe2db_vals": "单手近战/弓: (35-44)-(56-71) 火伤", "ring_matches": []},
  "大冰霜精华": {"poe2db_vals": "单手近战/弓: (31-38)-(47-59) 冰伤", "ring_matches": []},
  "大电光精华": {"poe2db_vals": "单手近战/弓: (1-6)-(85-107) 电伤", "ring_matches": []},
  "大磨蚀精华": {"poe2db_vals": "单手近战/弓: (16-24)-(28-42) 物伤", "ring_matches": []},
  "大迅捷精华": {"poe2db_vals": "法器/法杖: 施法速度25-28%", "ring_matches": []},
  "大魔法精华": {"poe2db_vals": "法器/法杖: 法术伤害75-89%", "ring_matches": []},
  "大急速精华": {"poe2db_vals": "近战: 攻速23-25%", "ring_matches": []},
  "大寻觅精华": {"poe2db_vals": "战斗武器: 暴击3.11-3.8%", "ring_matches": []},
  "大激战精华": {"poe2db_vals": "战斗/手套/箭袋: 命中237-346", "ring_matches": []},
  "大命令精华": {"poe2db_vals": "短杖: 友军伤害75-89%", "ring_matches": []},
}

for name, c in checks.items():
    print(f"\n--- {name} ---")
    print(f"  poe2db: {c['poe2db_vals']}")
    print(f"  => 戒指上{'无' if not c['ring_matches'] else '有对应'}词缀")
