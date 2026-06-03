# env.py

import numpy as np
import json, random
from typing import List, Optional, Tuple
from utils import GameData, Affix, ItemAction

OMEN_NONE = 0
OMEN_ANNUL_PREFIX = 1   # 左旋剥离: 下次移除只能移前缀
OMEN_ANNUL_SUFFIX = 2   # 右旋剥离: 下次移除只能移后缀
OMEN_LIGHT = 3          # 光明: 下次移除只能移除亵渎词缀
OMEN_EXALT_PREFIX = 4   # 左旋崇高: 下次添加只能加前缀
OMEN_EXALT_SUFFIX = 5   # 右旋崇高: 下次添加只能加后缀
OMEN_ESSENCE_PREFIX = 6 # 左旋结晶: 下次精华只能移除前缀
OMEN_ESSENCE_SUFFIX = 7 # 右旋结晶: 下次精华只能移除后缀

OMEN_PRICES = {
    OMEN_ANNUL_PREFIX: 200,
    OMEN_ANNUL_SUFFIX: 200,
    OMEN_LIGHT: 100,
    OMEN_EXALT_PREFIX: 500,
    OMEN_EXALT_SUFFIX: 500,
    OMEN_ESSENCE_PREFIX: 150,
    OMEN_ESSENCE_SUFFIX: 150,
}
OMEN_NAMES = {
    OMEN_ANNUL_PREFIX: "左旋剥离预兆",
    OMEN_ANNUL_SUFFIX: "右旋剥离预兆",
    OMEN_LIGHT: "光明预兆",
    OMEN_EXALT_PREFIX: "左旋崇高预兆",
    OMEN_EXALT_SUFFIX: "右旋崇高预兆",
    OMEN_ESSENCE_PREFIX: "左旋结晶预兆",
    OMEN_ESSENCE_SUFFIX: "右旋结晶预兆",
}


class GameEnv:
    def __init__(self, game_data: GameData, equipment_config: dict, reward_config: dict):
        self.game_data = game_data
        self.equipment_config = equipment_config
        self.reward_config = reward_config

        self.target_prefixes: List[str] = equipment_config["target_prefixes"]
        self.target_suffixes: List[str] = equipment_config["target_suffixes"]
        self.max_prefix: int = equipment_config.get("max_prefix", 3)
        self.max_suffix: int = equipment_config.get("max_suffix", 3)
        self.equipment_type: str = equipment_config.get("equipment_type", "ring")

        self.actions: List[ItemAction] = game_data.actions
        self.num_actions: int = len(self.actions)

        self.essence_map = {}
        self.desecrated_pool = {"prefixes": [], "suffixes": []}
        self._load_essences()
        self._load_desecrated()

        self.rarity: int = 0
        self.current_affixes: List[Affix] = []
        self.omen: int = OMEN_NONE
        self.steps_taken: int = 0
        self.budget_remaining: float = float(self.reward_config.get("max_budget", 5000))
        self.max_budget: float = float(self.reward_config.get("max_budget", 5000))
        self.max_tier: int = 12

        self.state_dim: int = 24
        self.reset()

    def _load_essences(self):
        try:
            with open("config/affixes/shared/essences.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, entry in data["essences"].items():
                self.essence_map[key] = entry
        except FileNotFoundError:
            pass

    def _load_desecrated(self):
        try:
            with open("config/affixes/sources/desecrated.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            for p in data.get("prefixes", []):
                self.desecrated_pool["prefixes"].append(p)
            for s in data.get("suffixes", []):
                self.desecrated_pool["suffixes"].append(s)
        except FileNotFoundError:
            pass

    def reset(self) -> np.ndarray:
        self.rarity = 0
        self.current_affixes = []
        self.omen = OMEN_NONE
        self.steps_taken = 0
        self.budget_remaining = self.max_budget
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        current_prefixes = [a for a in self.current_affixes if a.is_prefix]
        current_suffixes = [a for a in self.current_affixes if not a.is_prefix]
        num_prefixes = len(current_prefixes)
        num_suffixes = len(current_suffixes)

        current_prefix_names = {a.name for a in current_prefixes}
        current_suffix_names = {a.name for a in current_suffixes}
        satisfied_prefixes = sum(1 for t in self.target_prefixes if t in current_prefix_names)
        satisfied_suffixes = sum(1 for t in self.target_suffixes if t in current_suffix_names)
        goal_reached = (satisfied_prefixes == len(self.target_prefixes)
                        and satisfied_suffixes == len(self.target_suffixes))

        has_desecrated = 1.0 if any(getattr(a, "is_desecrated", False) for a in self.current_affixes) else 0.0

        omen_flags = np.zeros(4, dtype=np.float32)
        if self.omen == OMEN_ANNUL_PREFIX:
            omen_flags[0] = 1.0
        elif self.omen == OMEN_ANNUL_SUFFIX:
            omen_flags[1] = 1.0
        elif self.omen == OMEN_LIGHT:
            omen_flags[2] = 1.0
        elif self.omen == OMEN_EXALT_PREFIX:
            omen_flags[0] = 1.0
        elif self.omen == OMEN_EXALT_SUFFIX:
            omen_flags[1] = 1.0

        # 品质指标
        pqt = self._target_quality(True)
        sqt = self._target_quality(False)
        best, worst, avg_nt = self._affix_tier_metrics()
        target_names = set(self.target_prefixes) | set(self.target_suffixes)
        non_target_count = sum(1 for a in self.current_affixes if a.name not in target_names)

        state = np.array(
            [
                float(self.rarity),                             # [0]
                float(num_prefixes),                            # [1]
                float(num_suffixes),                            # [2]
                float(self.max_prefix - num_prefixes),          # [3]
                float(self.max_suffix - num_suffixes),          # [4]
                float(satisfied_prefixes),                      # [5]
                float(satisfied_suffixes),                      # [6]
                float(len(self.current_affixes)),               # [7]
                1.0 if goal_reached else 0.0,                    # [8]
                has_desecrated,                                 # [9]
                omen_flags[0],                                  # [10]
                omen_flags[1],                                  # [11]
                omen_flags[2],                                  # [12]
                self._cheapest_target_essence_price(),          # [13]
                0.0,                                            # [14] has_crafted_mod (future)
                self.budget_remaining / max(self.max_budget, 1),# [15]
                pqt,                                            # [16]
                sqt,                                            # [17]
                best,                                           # [18]
                worst,                                          # [19]
                avg_nt,                                         # [20]
                self.steps_taken / 50.0,                        # [21]
                1.0 if self.equipment_type == "ring" else 0.0,  # [22]
                float(non_target_count),                        # [23]
            ],
            dtype=np.float32,
        )
        return state

    def get_valid_actions(self) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=np.float32)
        if self.rarity == 0:
            allowed_max_prefix = 0
            allowed_max_suffix = 0
        elif self.rarity == 1:
            allowed_max_prefix = 1
            allowed_max_suffix = 1
        else:
            allowed_max_prefix = self.max_prefix
            allowed_max_suffix = self.max_suffix

        current_prefixes = len([a for a in self.current_affixes if a.is_prefix])
        current_suffixes = len([a for a in self.current_affixes if not a.is_prefix])
        current_total = len(self.current_affixes)
        can_add_prefix = current_prefixes < allowed_max_prefix
        can_add_suffix = current_suffixes < allowed_max_suffix
        has_room = can_add_prefix or can_add_suffix
        has_affixes = current_total > 0

        for action in self.actions:
            conds = action.conditions
            etype = action.effect.get("type", "")

            if "rarity" in conds and self.rarity != conds["rarity"]:
                continue
            if "rarity_min" in conds and self.rarity < conds["rarity_min"]:
                continue
            if conds.get("has_empty_slot", False) and not has_room:
                continue
            if "min_affix_count" in conds and current_total < conds["min_affix_count"]:
                continue
            if etype == "use_essence":
                if self._pick_best_essence() is None:
                    continue
            elif etype == "cheapest_essence":
                if self._pick_cheapest_essence() is None:
                    continue
            elif etype == "omen":
                if self.omen != OMEN_NONE:
                    continue
            else:
                if self.omen in (OMEN_LIGHT,) and etype == "remove_random_mod":
                    if not any(getattr(a, "is_desecrated", False) for a in self.current_affixes):
                        continue

            mask[action.id] = 1.0
        return mask

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        if action_idx < 0 or action_idx >= len(self.actions):
            return self._get_state(), -50.0, False, {"valid": False, "error": "Index Out of Range"}

        action = self.actions[action_idx]
        effect = action.effect
        etype = effect.get("type", "")

        current_mask = self.get_valid_actions()
        if current_mask[action_idx] == 0.0:
            return (self._get_state(), -10.0, False,
                    {"valid": False, "action_name": action.name, "error": "Illegal"})

        try:
            old_value = self._state_value()
            cost = float(action.price)
            reward_scale = float(self.reward_config.get("reward_scale", 100.0))
            success_bonus = float(self.reward_config.get("success_bonus", 10000.0))

            if etype == "add_random":
                self._handle_add_random(effect)
            elif etype == "reroll_single_mod":
                self._handle_reroll(effect)
            elif etype == "remove_random_mod":
                self._handle_remove()
            elif etype == "reroll_values":
                pass
            elif etype == "use_essence":
                cost += self._apply_best_essence()
            elif etype == "cheapest_essence":
                cost += self._apply_cheapest_essence()
            elif etype == "omen":
                self.omen = int(effect.get("omen_type", OMEN_NONE))
                cost += OMEN_PRICES.get(self.omen, 0)
            else:
                pass

            self.steps_taken += 1
            self.budget_remaining -= cost

            new_value = self._state_value()
            progress = (new_value - old_value) * reward_scale
            reward = -cost + progress

            state = self._get_state()
            done = bool(state[8] == 1.0)
            if done:
                reward += success_bonus
            elif self.budget_remaining <= 0 or self.steps_taken >= 50:
                done = True
                reward += float(self.reward_config.get("failure_penalty", -1000.0))

            return state, reward, done, {"valid": True, "action_name": action.name, "affixes_count": len(self.current_affixes)}

        except Exception as e:
            return self._get_state(), -50.0, False, {"valid": False, "error": str(e)}

    # --- 内部处理方法 ---

    def _handle_add_random(self, effect: dict):
        if "upgrade_rarity_to" in effect:
            self.rarity = effect["upgrade_rarity_to"]
        delta = effect.get("delta_affix_count", 1)
        min_ilvl = effect.get("min_ilvl", 1)
        for _ in range(delta):
            self._roll_and_add_affix(min_ilvl)
        # 消耗崇高预兆
        if self.omen in (OMEN_EXALT_PREFIX, OMEN_EXALT_SUFFIX):
            self.omen = OMEN_NONE

    def _handle_reroll(self, effect: dict):
        if self.current_affixes:
            self._remove_random_affix()
            self._roll_and_add_affix(min_ilvl=effect.get("min_ilvl", 1))

    def _handle_remove(self):
        """受 omen 影响的移除逻辑"""
        if not self.current_affixes:
            return

        if self.omen == OMEN_ANNUL_PREFIX:
            candidates = [a for a in self.current_affixes if a.is_prefix]
        elif self.omen == OMEN_ANNUL_SUFFIX:
            candidates = [a for a in self.current_affixes if not a.is_prefix]
        elif self.omen == OMEN_LIGHT:
            candidates = [a for a in self.current_affixes if getattr(a, "is_desecrated", False)]
        else:
            candidates = list(self.current_affixes)

        # 如果 omen 限定但找不到候选，退化为随机
        if not candidates:
            candidates = list(self.current_affixes)

        idx = random.randint(0, len(candidates) - 1)
        affix_to_remove = candidates[idx]
        self.current_affixes.remove(affix_to_remove)

        # 消耗预兆
        if self.omen in (OMEN_ANNUL_PREFIX, OMEN_ANNUL_SUFFIX, OMEN_LIGHT):
            self.omen = OMEN_NONE

    # ── 词缀品质指标 ──

    def _target_quality(self, is_prefix: bool) -> float:
        targets = self.target_prefixes if is_prefix else self.target_suffixes
        if not targets:
            return 0.0
        matching = [a for a in self.current_affixes if a.is_prefix == is_prefix]
        total = 0
        count = 0
        for t in targets:
            for a in matching:
                if a.name == t and a.tier > 0:
                    total += 1.0 - (a.tier / self.max_tier)
                    count += 1
                    break
        if count == 0:
            return 0.0
        return total / len(targets)

    def _affix_tier_metrics(self):
        tiers = [a.tier for a in self.current_affixes if a.tier > 0]
        if not tiers:
            return 0.0, 0.0, 0.0
        best = 1.0 - (min(tiers) / self.max_tier)
        worst = 1.0 - (max(tiers) / self.max_tier)
        target_names = set(self.target_prefixes) | set(self.target_suffixes)
        non_target_tiers = [a.tier for a in self.current_affixes if a.tier > 0 and a.name not in target_names]
        avg_nt = 1.0 - (sum(non_target_tiers) / max(len(non_target_tiers) * self.max_tier, 1)) if non_target_tiers else 0.5
        return best, worst, avg_nt

    def _state_value(self) -> float:
        pqt = self._target_quality(True)
        sqt = self._target_quality(False)
        target_names = set(self.target_prefixes) | set(self.target_suffixes)
        non_target = sum(1 for a in self.current_affixes if a.name not in target_names)
        return pqt + sqt - non_target * 0.05

    # ── 精华自动匹配系统 ──

    def _essence_matches_target(self, entry: dict) -> bool:
        slot_groups = entry.get("slot_groups", {})
        needed_prefixes = [t for t in self.target_prefixes
                           if t not in {a.name for a in self.current_affixes if a.is_prefix}]
        needed_suffixes = [t for t in self.target_suffixes
                           if t not in {a.name for a in self.current_affixes if not a.is_prefix}]
        if not needed_prefixes and not needed_suffixes:
            return False
        for sgrp_data in slot_groups.values():
            affix_id = sgrp_data.get("affix_id", "")
            possible = sgrp_data.get("possible_affixes", [affix_id] if affix_id else [])
            for a_id in possible:
                if a_id in needed_prefixes or a_id in needed_suffixes:
                    return True
        return False

    def _is_essence_usable(self, entry: dict) -> bool:
        etype = entry.get("type", "")
        if etype == "essence_normal":
            return self.rarity == 1 and self._has_empty_slot()
        if etype in ("essence_reroll", "essence_abyss"):
            return self.rarity == 2 and len(self.current_affixes) >= 1
        if etype == "essence_special":
            return self.rarity >= 1
        return False

    def _has_empty_slot(self) -> bool:
        pre_max = 1 if self.rarity == 1 else self.max_prefix
        suf_max = 1 if self.rarity == 1 else self.max_suffix
        pre = len([a for a in self.current_affixes if a.is_prefix])
        suf = len([a for a in self.current_affixes if not a.is_prefix])
        return pre < pre_max or suf < suf_max

    def _pick_best_essence(self):
        best = None
        for key, entry in self.essence_map.items():
            if not self._is_essence_usable(entry):
                continue
            if not self._essence_matches_target(entry):
                continue
            price = entry.get("price", 999)
            if best is None or price < best[1]:
                best = (key, price)
        return best

    def _pick_cheapest_essence(self):
        best = None
        for key, entry in self.essence_map.items():
            if not self._is_essence_usable(entry):
                continue
            price = entry.get("price", 999)
            if best is None or price < best[1]:
                best = (key, price)
        return best

    def _cheapest_target_essence_price(self) -> float:
        best = self._pick_best_essence()
        if best is None:
            return 0.0
        return best[1] / 100.0  # 归一化到 [0,1]

    def _apply_best_essence(self) -> float:
        best = self._pick_best_essence()
        if best is None:
            return 0.0
        return self._apply_essence_by_key(best[0])

    def _apply_cheapest_essence(self) -> float:
        best = self._pick_cheapest_essence()
        if best is None:
            return 0.0
        return self._apply_essence_by_key(best[0])

    def _apply_essence_by_key(self, essence_key: str) -> float:
        entry = self.essence_map.get(essence_key)
        if not entry:
            return 0.0

        etype = entry.get("type", "")
        price = entry.get("price", 0)
        reward = -price

        # essence_normal: 升级稀有度 + 添加词缀
        if etype == "essence_normal":
            if "upgrade_rarity_to" in entry.get("effect", {}):
                self.rarity = entry["effect"]["upgrade_rarity_to"]
            self._add_essence_affix(entry)

        # essence_reroll: 移除 + 添加
        elif etype == "essence_reroll":
            self._essence_remove_affix()
            self._add_essence_affix(entry)

        # essence_abyss: 移除 + 亵渎
        elif etype == "essence_abyss":
            self._essence_remove_affix()
            self._add_desecrated_affix()

        # essence_special: 品质/专属词缀, 不影响词缀计数
        elif etype == "essence_special":
            pass

        return reward

    def _essence_remove_affix(self):
        if not self.current_affixes:
            return
        if self.omen == OMEN_ESSENCE_PREFIX:
            cand = [a for a in self.current_affixes if a.is_prefix]
        elif self.omen == OMEN_ESSENCE_SUFFIX:
            cand = [a for a in self.current_affixes if not a.is_prefix]
        else:
            cand = list(self.current_affixes)
        if not cand:
            cand = list(self.current_affixes)
        idx = random.randint(0, len(cand) - 1)
        self.current_affixes.remove(cand[idx])
        if self.omen in (OMEN_ESSENCE_PREFIX, OMEN_ESSENCE_SUFFIX):
            self.omen = OMEN_NONE

    def _add_essence_affix(self, entry: dict):
        slot_groups = entry.get("slot_groups", {})
        affix_type_file = self.game_data.affix_file
        try:
            with open(affix_type_file, "r", encoding="utf-8") as f:
                eq_data = json.load(f)
        except FileNotFoundError:
            return
        for sgrp_data in slot_groups.values():
            affix_id = sgrp_data.get("affix_id", "")
            if not affix_id:
                continue
            p_affixes = eq_data.get("prefixes", [])
            s_affixes = eq_data.get("suffixes", [])
            found = None
            is_pref = True
            for a in p_affixes:
                if a["id"] == affix_id:
                    found = a; is_pref = True; break
            if not found:
                for a in s_affixes:
                    if a["id"] == affix_id:
                        found = a; is_pref = False; break
            if found:
                possible = sgrp_data.get("possible_affixes", [affix_id])
                chosen_id = random.choice(possible)
                if chosen_id != affix_id:
                    for a in p_affixes + s_affixes:
                        if a["id"] == chosen_id:
                            found = a; break
                lt = max(found["tiers"], key=lambda t: t["tier"])
                tier_val = lt.get("tier", 0) if isinstance(lt, dict) and "tier" in lt else 0
                new_affix = Affix(
                    name=chosen_id,
                    group=found["group"],
                    weight=lt["weight"],
                    is_prefix=is_pref,
                    tier=tier_val,
                )
                self.current_affixes.append(new_affix)
            break

    def _add_desecrated_affix(self):
        """从亵渎池添加一条随机亵渎词缀"""
        prefixes = self.desecrated_pool.get("prefixes", [])
        suffixes = self.desecrated_pool.get("suffixes", [])
        pool = prefixes + suffixes
        if not pool:
            return

        # 过滤同组冲突
        existing_groups = {a.group for a in self.current_affixes}
        valid = [d for d in pool if d["group"] not in existing_groups]
        if not valid:
            valid = pool  # 防死锁

        chosen = random.choices(valid, weights=[d.get("weight", 1000) for d in valid], k=1)[0]
        new_affix = Affix(
            name=chosen["id"],
            group=chosen["group"],
            weight=chosen.get("weight", 1000),
            is_prefix=chosen in prefixes,
            tier=0,
        )
        new_affix.is_desecrated = True
        self.current_affixes.append(new_affix)

    def _roll_and_add_affix(self, min_ilvl: int = 1):
        prefixes_len = len([a for a in self.current_affixes if a.is_prefix])
        suffixes_len = len([a for a in self.current_affixes if not a.is_prefix])
        can_prefix = prefixes_len < self.max_prefix
        can_suffix = suffixes_len < self.max_suffix
        if not can_prefix and not can_suffix:
            return

        # 预兆影响: 崇高预兆强制前缀/后缀
        if self.omen == OMEN_EXALT_PREFIX:
            roll_prefix = True
        elif self.omen == OMEN_EXALT_SUFFIX:
            roll_prefix = False
        else:
            roll_prefix = random.choice([True, False]) if (can_prefix and can_suffix) else can_prefix

        # 从 per-equipment-type 数据加载词缀池
        try:
            with open("config/affixes/equipment_types/ring.json", "r", encoding="utf-8") as f:
                ring_data = json.load(f)
        except FileNotFoundError:
            # fallback 到旧的 sample_affix
            new_affix = self.game_data.sample_affix(self.current_affixes, roll_prefix)
            if new_affix:
                self.current_affixes.append(new_affix)
            return

        pool = ring_data["prefixes"] if roll_prefix else ring_data["suffixes"]
        existing_groups = {a.group for a in self.current_affixes}

        # 展平所有(tier, weight)对
        flat = []
        for affix_entry in pool:
            if affix_entry["group"] in existing_groups:
                continue
            total_w = sum(t["weight"] for t in affix_entry["tiers"] if t["ilvl"] <= min_ilvl or t["tier"] == 1)
            # 简化的 ilvl 过滤: 取所有 ilvl <= min_ilvl 的 tier
            valid_tiers = [t for t in affix_entry["tiers"] if t["ilvl"] <= min_ilvl]
            if not valid_tiers:
                valid_tiers = [affix_entry["tiers"][-1]]  # 最低级保底
            flat.append((affix_entry, valid_tiers))

        if not flat:
            return

        chosen_entry, chosen_tiers = random.choices(
            flat,
            weights=[sum(t["weight"] for t in ts) for _, ts in flat],
            k=1
        )[0]
        tier = random.choices(chosen_tiers, weights=[t["weight"] for t in chosen_tiers], k=1)[0]

        new_affix = Affix(
            name=chosen_entry["id"],
            group=chosen_entry["group"],
            weight=tier["weight"],
            is_prefix=roll_prefix,
            tier=tier["tier"],
        )
        self.current_affixes.append(new_affix)

    def _remove_random_affix(self):
        if self.current_affixes:
            idx_to_remove = random.randint(0, len(self.current_affixes) - 1)
            self.current_affixes.pop(idx_to_remove)

    def render(self) -> str:
        rarity_names = {0: "白装", 1: "魔法", 2: "稀有"}
        info = f"稀有度: {rarity_names.get(self.rarity, '未知')}\n"
        if self.omen != OMEN_NONE:
            info += f"预兆: {OMEN_NAMES.get(self.omen, f'未知({self.omen})')}\n"

        if self.rarity == 0:
            current_max_p, current_max_s = 0, 0
        elif self.rarity == 1:
            current_max_p, current_max_s = 1, 1
        else:
            current_max_p, current_max_s = self.max_prefix, self.max_suffix

        prefixes = [a for a in self.current_affixes if a.is_prefix]
        suffixes = [a for a in self.current_affixes if not a.is_prefix]

        info += f"  前缀 ({len(prefixes)}/{current_max_p}):\n"
        for i, p in enumerate(prefixes):
            des = " [亵渎]" if getattr(p, "is_desecrated", False) else ""
            info += f"    {i+1}. {p.name} (group: {p.group}){des}\n"
        info += f"  后缀 ({len(suffixes)}/{current_max_s}):\n"
        for i, s in enumerate(suffixes):
            des = " [亵渎]" if getattr(s, "is_desecrated", False) else ""
            info += f"    {i+1}. {s.name} (group: {s.group}){des}\n"
        return info
