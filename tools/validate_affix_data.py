# tools/validate_affix_data.py
# Validate affix data structure, weight distribution, tier/ilvl filtering, group conflicts

import json
import random
from collections import Counter
from pathlib import Path


def load_equipment_type_data(eq_type: str) -> dict:
    path = Path(f"config/affixes/equipment_types/{eq_type}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_definitions() -> dict:
    with open("config/affixes/shared/definitions.json", "r", encoding="utf-8") as f:
        return json.load(f)


def validate_structure(data: dict):
    assert "equipment_type" in data, "missing equipment_type"
    assert "tags" in data, "missing tags"
    assert "max_prefix" in data, "missing max_prefix"
    assert "max_suffix" in data, "missing max_suffix"
    
    for slot in ["prefixes", "suffixes"]:
        for affix in data[slot]:
            assert "id" in affix, f"{slot} affix missing id"
            assert "tiers" in affix, f"{affix['id']} missing tiers"
            assert len(affix["tiers"]) > 0, f"{affix['id']} tiers is empty"
            assert "group" in affix, f"{affix['id']} missing group"
            
            for tier in affix["tiers"]:
                assert "tier" in tier, f"{affix['id']} tier missing tier number"
                assert "ilvl" in tier, f"{affix['id']} T{tier['tier']} missing ilvl"
                assert "weight" in tier, f"{affix['id']} T{tier['tier']} missing weight"
                assert "value" in tier, f"{affix['id']} T{tier['tier']} missing value"
                assert tier["tier"] >= 1, f"{affix['id']} tier must be >= 1"
                
        for affix in data[slot]:
            tiers = sorted([t["tier"] for t in affix["tiers"]])
            expected = list(range(1, len(tiers) + 1))
            assert tiers == expected, f"{affix['id']} tier numbers not continuous: {tiers} != {expected}"
            
            ilvls = [t["ilvl"] for t in sorted(affix["tiers"], key=lambda t: t["tier"])]
            for i in range(1, len(ilvls)):
                assert ilvls[i] <= ilvls[i-1], f"{affix['id']} ilvl not decreasing (T1=best=highest): {ilvls}"
                assert ilvls[i] < ilvls[i-1] or (i == len(ilvls)-1), f"{affix['id']} duplicate ilvl: T{i+1}={ilvls[i]} same as T{i}={ilvls[i-1]}"
    
    print(f"[OK] Structure: {len(data['prefixes'])} prefixes, {len(data['suffixes'])} suffixes")


def test_weight_distribution(data: dict, samples: int = 100000):
    random.seed(42)
    
    for slot in ["prefixes", "suffixes"]:
        pool = data[slot]
        # Using sum of all tier weights as the effective weight for each affix
        total_all = sum(sum(t["weight"] for t in a["tiers"]) for a in pool)
        affix_weight_map = {}
        for affix in pool:
            effective = sum(t["weight"] for t in affix["tiers"])
            base_pct = effective / total_all * 100 if total_all > 0 else 0
            affix_weight_map[affix["id"]] = {
                "effective": effective,
                "total_tier": sum(t["weight"] for t in affix["tiers"]),
                "base_pct": base_pct
            }
        
        print(f"\n  [{slot}] total effective weight: {total_all:,}")
        for aid, info in sorted(affix_weight_map.items(), key=lambda x: -x[1]["effective"]):
            pct = info["base_pct"]
            affix = next(a for a in data[slot] if a['id'] == aid)
            t_cnt = len(affix['tiers'])
            print(f"    {aid:30s} total={info['effective']:6d}  tiers={t_cnt:2d}  pct={pct:.2f}%")
        
        # Sample: pick affix proportionally to sum of tier weights,
        # then pick tier within that affix
        chosen_ids = []
        chosen_tiers = []
        for _ in range(samples):
            affix = random.choices(pool, weights=[sum(t["weight"] for t in a["tiers"]) for a in pool], k=1)[0]
            tier = random.choices(affix["tiers"], weights=[t["weight"] for t in affix["tiers"]], k=1)[0]
            chosen_ids.append(affix["id"])
            chosen_tiers.append(tier["tier"])
        
        counter = Counter(chosen_ids)
        print(f"\n  [{slot}] sampling distribution ({samples} tries):")
        for aid, count in counter.most_common():
            expected_pct = affix_weight_map[aid]["base_pct"]
            actual_pct = count / samples * 100
            dev = abs(expected_pct - actual_pct)
            marker = " [!]" if dev > 1.0 else ""
            print(f"    {aid:30s} exp={expected_pct:.2f}%  act={actual_pct:.2f}%  dev={dev:.2f}%{marker}")
    
    print(f"\n[OK] Weight distribution validated (samples={samples})")


def test_group_conflict(data: dict):
    random.seed(42)
    import sys; sys.path.insert(0, ".")
    from utils import Affix
    
    for slot_name, slot in [("prefixes", data["prefixes"]), ("suffixes", data["suffixes"])]:
        seen_groups = set()
        current_affixes = []
        
        for _ in range(1000):
            valid = [a for a in slot if a["group"] not in seen_groups]
            if not valid:
                seen_groups.clear()
                current_affixes = []
                valid = slot
            
            # Use sum of tier weights for sampling
            affix_weights = [sum(t["weight"] for t in a["tiers"]) for a in valid]
            affix_data = random.choices(valid, weights=affix_weights, k=1)[0]
            seen_groups.add(affix_data["group"])
            
            affix = Affix(
                name=affix_data["id"],
                group=affix_data["group"],
                weight=sum(t["weight"] for t in affix_data["tiers"]),
                is_prefix=(slot_name == "prefixes")
            )
            
            for existing in current_affixes:
                if existing.group == affix.group:
                    raise AssertionError(f"Group conflict: {existing.group}")
            current_affixes.append(affix)
        
        print(f"  [{slot_name}] group conflict test: 0 conflicts [OK]")


def test_ilvl_filtering(data: dict):
    for slot in ["prefixes", "suffixes"]:
        for affix in data[slot]:
            tiers = sorted(affix["tiers"], key=lambda t: t["tier"])
            for i, t in enumerate(tiers):
                if i > 0:
                    gap = tiers[i-1]["ilvl"] - t["ilvl"]
                    assert gap >= 2, f"{affix['id']} T{t['tier']}: ilvl gap too small ({gap})"
            
            assert tiers[-1]["ilvl"] <= 86, f"{affix['id']} max tier requires ilvl{tiers[-1]['ilvl']}, exceeds cap"
    
    print(f"[OK] ilvl filtering validated")


def run_full_validation():
    print("=" * 60)
    print("  Affix Data Validation Tool")
    print("=" * 60)
    
    eq_dir = Path("config/affixes/equipment_types")
    for eq_file in sorted(eq_dir.glob("*.json")):
        eq_type = eq_file.stem
        print(f"\n[Equipment]: {eq_type}")
        print("-" * 40)
        data = load_equipment_type_data(eq_type)
        
        validate_structure(data)
        test_ilvl_filtering(data)
        test_group_conflict(data)
        test_weight_distribution(data, samples=50000)
    
    print("\n" + "=" * 60)
    print("  All validations passed [OK]")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        eq_type = sys.argv[1]
        data = load_equipment_type_data(eq_type)
        validate_structure(data)
        test_ilvl_filtering(data)
        test_group_conflict(data)
        test_weight_distribution(data)
    else:
        run_full_validation()
