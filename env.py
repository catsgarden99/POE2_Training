# env.py

import numpy as np
from typing import List, Optional, Tuple
from utils import GameData, Affix, ItemAction


class GameEnv:
    """
    《流放之路2》做装强化学习环境

    管理一件装备从白装到成品的完整做装过程，支持：
    - 真实的装备状态管理（稀有度、前后缀数量、同组互斥）
    - 动作掩码（根据游戏规则动态计算合法动作）
    - 目标达成检测
    """

    def __init__(
        self,
        game_data: GameData,
        equipment_config: dict,
        reward_config: dict,
    ):
        """
        初始化做装环境
        """
        self.game_data = game_data
        self.equipment_config = equipment_config
        self.reward_config = reward_config

        # 目标词缀配置
        self.target_prefixes: List[str] = equipment_config["target_prefixes"]
        self.target_suffixes: List[str] = equipment_config["target_suffixes"]

        # 词缀数量限制
        self.max_prefix: int = equipment_config.get("max_prefix", 3)
        self.max_suffix: int = equipment_config.get("max_suffix", 3)

        # 动作空间
        self.actions: List[ItemAction] = game_data.actions
        self.num_actions: int = len(self.actions)

        # 装备状态初始化
        self.rarity: int = 0  # 0=白装, 1=魔法, 2=稀有
        self.current_affixes: List[Affix] = []

        # 状态向量维度
        self.state_dim: int = 9
        
        # 强制安全重置
        self.reset()

    def reset(self) -> np.ndarray:
        """
        重置环境：清空装备，回到白装状态
        """
        self.rarity = 0
        self.current_affixes = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        生成当前装备状态的特征向量 (9维)
        """
        # 统计当前前后缀
        current_prefixes = [a for a in self.current_affixes if a.is_prefix]
        current_suffixes = [a for a in self.current_affixes if not a.is_prefix]

        num_prefixes = len(current_prefixes)
        num_suffixes = len(current_suffixes)

        # 检查目标达成情况
        current_prefix_names = {a.name for a in current_prefixes}
        current_suffix_names = {a.name for a in current_suffixes}

        satisfied_prefixes = sum(
            1 for target in self.target_prefixes if target in current_prefix_names
        )
        satisfied_suffixes = sum(
            1 for target in self.target_suffixes if target in current_suffix_names
        )

        # 判断是否达成最终目标 (必须完美包含所有勾选的目标词缀)
        all_prefixes_met = satisfied_prefixes == len(self.target_prefixes)
        all_suffixes_met = satisfied_suffixes == len(self.target_suffixes)
        goal_reached = all_prefixes_met and all_suffixes_met

        state = np.array(
            [
                float(self.rarity),                        # [0] 稀有度
                float(num_prefixes),                      # [1] 当前前缀数
                float(num_suffixes),                      # [2] 当前后缀数
                float(self.max_prefix - num_prefixes),    # [3] 剩余前缀空位
                float(self.max_suffix - num_suffixes),    # [4] 剩余后缀空位
                float(satisfied_prefixes),                # [5] 已满足目标前缀数
                float(satisfied_suffixes),                # [6] 已满足目标后缀数
                float(len(self.current_affixes)),         # [7] 总词缀数
                1.0 if goal_reached else 0.0,             # [8] 是否达成目标
            ],
            dtype=np.float32,
        )
        return state

    def get_valid_actions(self) -> np.ndarray:
        """
        根据当前装备状态，计算所有动作的掩码 (0/1)
        """
        valid_mask = np.ones(self.num_actions, dtype=np.int32)

        # 获取当前词缀统计
        num_prefixes = len([a for a in self.current_affixes if a.is_prefix])
        num_suffixes = len([a for a in self.current_affixes if not a.is_prefix])
        no_affixes = len(self.current_affixes) == 0

        for idx, action in enumerate(self.actions):
            # 1. 检查稀有度要求
            if "rarity" in action.conditions:
                required_rarity = action.conditions["rarity"]
                is_valid_rarity = False
                if required_rarity == self.rarity:
                    is_valid_rarity = True
                
                # 兼容点金石或类似蜕变动作的特殊转移
                if action.name == "点金石" and self.rarity == 0:
                    is_valid_rarity = True

                if not is_valid_rarity:
                    valid_mask[idx] = 0
                    continue

            # 2. 检查词缀数量限制
            effect_type = action.effect.get("type", "")
            if effect_type == "add_random":
                adds_prefix = action.effect.get("adds_prefix", False)
                adds_suffix = action.effect.get("adds_suffix", False)

                if adds_prefix and num_prefixes >= self.max_prefix:
                    valid_mask[idx] = 0
                    continue
                if adds_suffix and num_suffixes >= self.max_suffix:
                    valid_mask[idx] = 0
                    continue

            elif effect_type == "remove_random":
                if no_affixes:
                    valid_mask[idx] = 0
                    continue

            elif effect_type == "reroll_all":
                if self.rarity == 0:  # 白装不能被重铸
                    valid_mask[idx] = 0
                    continue

            # 3. 特殊通货名称硬阻断
            if "chaos" in action.name.lower() and self.rarity != 2:
                valid_mask[idx] = 0
                continue

        return valid_mask

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一个做装动作
        """
        valid_mask = self.get_valid_actions()
        if valid_mask[action_idx] == 0:
            return (self._get_state(), -100.0, False, {"valid": False})

        action = self.actions[action_idx]
        reward = -float(action.price)
        done = False

        effect = action.effect
        effect_type = effect.get("type", "")

        try:
            if effect_type == "add_random":
                roll_prefix = effect.get("adds_prefix", False)
                roll_suffix = effect.get("adds_suffix", False)

                if roll_prefix and roll_suffix:
                    roll_prefix = np.random.random() < 0.5
                    roll_suffix = not roll_prefix

                if roll_prefix:
                    new_affix = self.game_data.sample_affix(self.current_affixes, roll_prefix=True)
                    if self.rarity == 0:
                        self.rarity = 1
                else:
                    new_affix = self.game_data.sample_affix(self.current_affixes, roll_prefix=False)
                    if self.rarity == 0:
                        self.rarity = 1

                if new_affix is not None:
                    self.current_affixes.append(new_affix)
                    if len(self.current_affixes) >= 3 and self.rarity < 2:
                        self.rarity = 2

            elif effect_type == "remove_random":
                if self.current_affixes:
                    remove_idx = np.random.randint(len(self.current_affixes))
                    self.current_affixes.pop(remove_idx)

            elif effect_type == "reroll_all":
                self.current_affixes = []
                if self.rarity >= 1:
                    target_count = np.random.randint(1, 7) if self.rarity >= 2 else np.random.randint(1, 3)
                    for _ in range(target_count):
                        roll_p = np.random.random() < 0.5
                        new_affix = self.game_data.sample_affix(self.current_affixes, roll_prefix=roll_p)
                        if new_affix:
                            self.current_affixes.append(new_affix)

            state = self._get_state()
            if state[8] == 1.0:
                reward += float(self.reward_config["success_bonus"])
                done = True

            return (
                state,
                reward,
                done,
                {
                    "valid": True,
                    "action_name": action.name,
                    "affixes_count": len(self.current_affixes),
                },
            )

        except Exception as e:
            return (
                self._get_state(),
                -50.0,
                False,
                {"valid": False, "error": str(e)},
            )

    def render(self) -> str:
        """
        打印当前装备状态
        """
        rarity_names = {0: "白装", 1: "魔法", 2: "稀有"}
        info = f"稀有度: {rarity_names.get(self.rarity, '未知')}\n"
        
        prefixes = [a for a in self.current_affixes if a.is_prefix]
        suffixes = [a for a in self.current_affixes if not a.is_prefix]

        info += f"  前缀 ({len(prefixes)}/{self.max_prefix}):\n"
        for i, p in enumerate(prefixes):
            info += f"    {i+1}. {p.name} (group: {p.group})\n"

        info += f"  后缀 ({len(suffixes)}/{self.max_suffix}):\n"
        for i, s in enumerate(suffixes):
            info += f"    {i+1}. {s.name} (group: {s.group})\n"

        return info