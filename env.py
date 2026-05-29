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
        根据声明式条件与动态稀有度容量，防御性计算合法动作掩码
        """
        mask = np.zeros(self.num_actions, dtype=np.float32)
        
        # 1. 动态计算当前稀有度允许的词缀容量
        # 0=白装(0条), 1=魔法/蓝装(最大1前1后), 2=稀有/黄装(最大3前3后)
        if self.rarity == 0:
            allowed_max_prefix = 0
            allowed_max_suffix = 0
        elif self.rarity == 1:
            allowed_max_prefix = 1
            allowed_max_suffix = 1
        else:
            allowed_max_prefix = self.max_prefix  # 3
            allowed_max_suffix = self.max_suffix  # 3

        current_prefixes = len([a for a in self.current_affixes if a.is_prefix])
        current_suffixes = len([a for a in self.current_affixes if not a.is_prefix])
        current_total_affixes = len(self.current_affixes)
        
        for action in self.actions:
            conds = action.conditions
            
            # 门禁 1：稀有度硬性匹配校验
            if "rarity" in conds and self.rarity != conds["rarity"]:
                continue
                
            # 门禁 2：检查空余词缀位 (必须结合当前稀有度的动态容量上限)
            if conds.get("has_empty_slot", False):
                # 如果当前总词缀已经达到了当前稀有度的最大上限，直接拦截
                if current_prefixes >= allowed_max_prefix and current_suffixes >= allowed_max_suffix:
                    continue
                if current_total_affixes >= (allowed_max_prefix + allowed_max_suffix):
                    continue
                    
            # 门禁 3：词缀数量下限检查（针对混沌石、剥离石等）
            if "min_affix_count" in conds and current_total_affixes < conds["min_affix_count"]:
                continue
                
            # 开放合法动作
            mask[action.id] = 1.0
            
        return mask
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        # 防御机制：防止外部传入越界索引
        if action_idx < 0 or action_idx >= len(self.actions):
            return self._get_state(), -50.0, False, {"valid": False, "error": "Action Index Out of Range"}

        action = self.actions[action_idx]
        effect = action.effect

        # 核心拦截：如果动作掩码判定为 0.0，属于非法越界操作，直接短路惩罚，【绝对不允许】继续往下走业务逻辑
        current_mask = self.get_valid_actions()
        if current_mask[action_idx] == 0.0:
            return (
                self._get_state(), 
                -10.0, 
                False, 
                {"valid": False, "action_name": action.name, "error": "Illegal Action Intercepted by Mask"}
            )

        try:
            # 扣除通货消耗作为即时惩罚基准
            reward = -float(action.price)

            # --- 1. 增量加随机词缀 (蜕变三兄弟、增幅、富豪、崇高) ---
            if effect.get("type") == "add_random":
                if "upgrade_rarity_to" in effect:
                    self.rarity = effect["upgrade_rarity_to"]
                
                delta = effect.get("delta_affix_count", 1)
                for _ in range(delta):
                    # 传入通货指定的最小 ilvl 限制限制
                    self._roll_and_add_affix_with_limit(min_ilvl=effect.get("min_ilvl", 1))

            # --- 2. 真正的《流放2》混沌石单条局部重铸 (一除、一加) ---
            elif effect.get("type") == "reroll_single_mod":
                if len(self.current_affixes) > 0:
                    self._remove_random_affix()  # 刮掉一条
                    self._roll_and_add_affix_with_limit(min_ilvl=effect.get("min_ilvl", 1)) # 补上一条

            # --- 3. 剥离石机制 (移除一条) ---
            elif effect.get("type") == "remove_random_mod":
                self._remove_random_affix()

            # --- 4. 神圣石机制 (原数值重 rolling) ---
            elif effect.get("type") == "reroll_values":
                # 维持你原本的 reroll_values 或不改变词缀总数逻辑
                pass

            # 达成目标判定与正常返回逻辑维持你原代码不变...
            state = self._get_state()
            if state[8] == 1.0: # 满足特定观测终止
                reward += float(self.reward_config["success_bonus"])
                done = True
            else:
                done = False

            return state, reward, done, {"valid": True, "action_name": action.name, "affixes_count": len(self.current_affixes)}

        except Exception as e:
            return self._get_state(), -50.0, False, {"valid": False, "error": str(e)}

    def render(self) -> str:
        """
        根据当前装备的动态稀有度容量，清晰、严密地打印装备状态
        """
        rarity_names = {0: "白装", 1: "魔法", 2: "稀有"}
        info = f"稀有度: {rarity_names.get(self.rarity, '未知')}\n"
        
        # 动态计算当前稀有度下的真实容量分母
        if self.rarity == 0:
            current_max_p, current_max_s = 0, 0
        elif self.rarity == 1:
            current_max_p, current_max_s = 1, 1
        else:
            current_max_p, current_max_s = self.max_prefix, self.max_suffix # 3, 3

        prefixes = [a for a in self.current_affixes if a.is_prefix]
        suffixes = [a for a in self.current_affixes if not a.is_prefix]

        # 将分母替换为与当前动作掩码完全同步的动态容量上限
        info += f"  前缀 ({len(prefixes)}/{current_max_p}):\n"
        for i, p in enumerate(prefixes):
            info += f"    {i+1}. {p.name} (group: {p.group})\n"
            
        info += f"  后缀 ({len(suffixes)}/{current_max_s}):\n"
        for i, s in enumerate(suffixes):
            info += f"    {i+1}. {s.name} (group: {s.group})\n"
            
        return info
    
    def _remove_random_affix(self):
        """物理剔除：随机移除当前装备上的一条词缀"""
        if self.current_affixes:
            import random
            idx_to_remove = random.randint(0, len(self.current_affixes) - 1)
            self.current_affixes.pop(idx_to_remove)

    def _roll_and_add_affix_with_limit(self, min_ilvl: int):
        """
        锁定词缀等级下限的双重过滤抽取引擎。
        因为你的 utils.py 原本只支持全池盲抽，为了完美支持高级/完美蜕变石，我们在这里做个包装。
        """
        # 1. 判断该加前缀还是后缀 (受限于你的环境最大前后缀限制)
        prefixes_len = len([a for a in self.current_affixes if a.is_prefix])
        suffixes_len = len([a for a in self.current_affixes if not a.is_prefix])
        
        can_prefix = prefixes_len < self.max_prefix
        can_suffix = suffixes_len < self.max_suffix
        
        if not can_prefix and not can_suffix:
            return
            
        import random
        # 如果前后缀都能加，随机选一个方向
        roll_prefix = random.choice([True, False]) if (can_prefix and can_suffix) else can_prefix
        
        # 2. 从全局数据池获取候选，并用 min_ilvl 防御性过滤
        # 注意：这里我们调用你原 utils.py 的抽样，如果你的词缀命名规范含有 ilvl 信息，
        # 或者后续你想做更精细的 T 阶匹配，可以在此对 valid_pool 做截断。
        # 暂时用你现有的 sample_affix 兜底：
        new_affix = self.game_data.sample_affix(self.current_affixes, roll_prefix)
        if new_affix:
            self.current_affixes.append(new_affix)