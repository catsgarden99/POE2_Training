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

        self.actions: List[ItemAction] = game_data.actions
        self.num_actions: int = len(self.actions)

        # 加载精华映射表和亵渎词缀池
        self.essence_map = {}
        self.desecrated_pool = {"prefixes": [], "suffixes": []}
        self._load_essences()
        self._load_desecrated()

        self.rarity: int = 0
        self.current_affixes: List[Affix] = []
        self.omen: int = OMEN_NONE  # 当前激活的预兆

        # 状态: 9原始 + 4预兆标记 = 13维
        self.state_dim: int = 13
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

        # [9] ~ [12]: 4 个预兆标记
        omen_flags = np.zeros(4, dtype=np.float32)
        if self.omen == OMEN_ANNUL_PREFIX:
            omen_flags[0] = 1.0  # 移除限定前缀
        elif self.omen == OMEN_ANNUL_SUFFIX:
            omen_flags[1] = 1.0  # 移除限定后缀
        elif self.omen == OMEN_LIGHT:
            omen_flags[2] = 1.0  # 移除限定亵渎
        elif self.omen == OMEN_EXALT_PREFIX:
            omen_flags[0] = 1.0  # 复用: 添加限定前缀
        elif self.omen == OMEN_EXALT_SUFFIX:
            omen_flags[1] = 1.0  # 复用: 添加限定后缀

        state = np.array(
            [
                float(self.rarity),                     # [0]
                float(num_prefixes),                    # [1]
                float(num_suffixes),                    # [2]
                float(self.max_prefix - num_prefixes),  # [3]
                float(self.max_suffix - num_suffixes),  # [4]
                float(satisfied_prefixes),              # [5]
                float(satisfied_suffixes),              # [6]
                float(len(self.current_affixes)),       # [7]
                1.0 if goal_reached else 0.0,           # [8]
                has_desecrated,                         # [9]
                omen_flags[0],                          # [10] sinistral
                omen_flags[1],                          # [11] dextral
                omen_flags[2],                          # [12] light
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
            if conds.get("has_empty_slot", False) and not has_room:
                continue
            if "min_affix_count" in conds and current_total < conds["min_affix_count"]:
                continue
            if etype == "essence_add":
                # 精华只能用在稀有或魔法物品上, 需要有对应空位
                if self.rarity == 0 or not has_room:
                    continue
            if etype == "essence_abyss":
                if self.rarity == 0 or not has_affixes:
                    continue
            if etype == "omen":
                # 不能同时激活两个预兆
                if self.omen != OMEN_NONE:
                    continue
            else:
                # 如果 omen 限定移除方向, 检查当前是否符合条件
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
            reward = -float(action.price)

            if etype == "add_random":
                self._handle_add_random(effect)
            elif etype == "reroll_single_mod":
                self._handle_reroll(effect)
            elif etype == "remove_random_mod":
                self._handle_remove(action)
            elif etype == "reroll_values":
                pass
            elif etype == "essence_add":
                reward += self._handle_essence_add(action)
            elif etype == "essence_abyss":
                reward += self._handle_essence_abyss()
            elif etype == "omen":
                self.omen = int(effect.get("omen_type", OMEN_NONE))
                reward -= OMEN_PRICES.get(self.omen, 0)
            else:
                pass

            state = self._get_state()
            done = bool(state[8] == 1.0)
            if done:
                reward += float(self.reward_config["success_bonus"])

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

    def _handle_remove(self, action: ItemAction):
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

    def _handle_essence_add(self, action: ItemAction) -> float:
        """添加精华指定的词缀(固定 tier, 固定 group)"""
        essence_key = action.effect.get("essence_key", "")
        entry = self.essence_map.get(essence_key)
        if not entry:
            return 0.0

        grants = entry.get("grants", {})
        affix_id = grants.get("affix_id", "")
        tier_target = grants.get("tier", 1)
        affix_group = grants.get("affix_group", "")

        # 从 ring.json 找对应的词缀数据
        affix_data = None
        with open("config/affixes/equipment_types/ring.json", "r", encoding="utf-8") as f:
            ring_data = json.load(f)

        for pool, is_pref in [(ring_data["prefixes"], True), (ring_data["suffixes"], False)]:
            for a in pool:
                if affix_id and a["id"] == affix_id:
                    affix_data = a
                    is_prefix = is_pref
                    break
                if affix_group and affix_group in a.get("group", ""):
                    # 永恒精华类: 给随机属性
                    affix_data = a
                    is_prefix = is_pref
                    break

        if affix_data:
            # 找到对应的 tier
            for t in affix_data["tiers"]:
                if t["tier"] == tier_target:
                    new_affix = Affix(
                        name=affix_id or affix_data["id"],
                        group=affix_data["group"],
                        weight=t["weight"],
                        is_prefix=is_prefix,
                    )
                    self.current_affixes.append(new_affix)
                    break

        # 如果 omen 限定精华移除方向, 在精华使用前先移除一条
        if self.omen in (OMEN_ESSENCE_PREFIX, OMEN_ESSENCE_SUFFIX):
            if self.current_affixes:
                self._handle_remove(action)
            self.omen = OMEN_NONE

        return 0.0

    def _handle_essence_abyss(self) -> float:
        """深渊精华: 移除一条随机词缀 + 添加亵渎词缀"""
        if self.current_affixes:
            # 受 omen 影响的移除
            if self.omen == OMEN_ESSENCE_PREFIX:
                cand = [a for a in self.current_affixes if a.is_prefix]
                if cand:
                    idx = random.randint(0, len(cand) - 1)
                    self.current_affixes.remove(cand[idx])
                self.omen = OMEN_NONE
            elif self.omen == OMEN_ESSENCE_SUFFIX:
                cand = [a for a in self.current_affixes if not a.is_prefix]
                if cand:
                    idx = random.randint(0, len(cand) - 1)
                    self.current_affixes.remove(cand[idx])
                self.omen = OMEN_NONE
            else:
                self._remove_random_affix()

        # 添加亵渎词缀 (从 desecrated 池按权重抽)
        self._add_desecrated_affix()
        return 0.0

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
