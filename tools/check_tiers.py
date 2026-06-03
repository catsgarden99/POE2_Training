import json
ring = json.load(open('config/affixes/equipment_types/ring.json', 'r', encoding='utf-8'))
targets = ['physical_damage','fire_damage','cold_damage','lightning_damage','cast_speed','item_rarity','maximum_mana','maximum_life','strength','dexterity','intelligence','all_attributes']
for a in ring['prefixes'] + ring['suffixes']:
    if a['id'] in targets:
        for t in a['tiers']:
            v = t.get('value', '?')
            print(f"{a['id']} T{t['tier']} val={v} weight={t['weight']}")
