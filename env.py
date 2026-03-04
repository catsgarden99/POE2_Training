import numpy as np
import random
from utils import GameData

class GameEnv:
    """游戏环境，模拟装备制作过程"""
    def __init__(self, game_data: GameData, equipment_config: dict, reward_config: dict):
        """
        :param game_data: GameData 实例，包含道具和词缀数据
        :param equipment_config: 装备配置字典（从 JSON 加载）
        :param reward_config: 奖励配置字典（从 training.json 加载）
        """
        self.data = game_data
        self.equip = equipment_config
        self.reward_config = reward_config
        self.actions = game_data.actions
        self.num_actions = len(self.actions)

        # 从装备配置中读取最大词缀数（稀有物品的上限）
        self.max_prefix = self.equip['max_prefix']
        self.max_suffix = self.equip['max_suffix']
        self.target_prefix_pool = set(self.equip['target_prefixes'])
        self.target_suffix_pool = set(self.equip['target_suffixes'])

        self.reset()

    # ---------- 辅助方法：根据稀有度返回最大词缀数 ----------
    def _max_prefix_for_rarity(self, rarity: int) -> int:
        """给定稀有度，返回允许的最大前缀数"""
        if rarity == 0:
            return 0
        elif rarity == 1:
            return 1
        else:  # 稀有
            return self.max_prefix

    def _max_suffix_for_rarity(self, rarity: int) -> int:
        """给定稀有度，返回允许的最大后缀数"""
        if rarity == 0:
            return 0
        elif rarity == 1:
            return 1
        else:
            return self.max_suffix

    def _max_prefix(self) -> int:
        """当前稀有度下的最大前缀数"""
        return self._max_prefix_for_rarity(self.rarity)

    def _max_suffix(self) -> int:
        """当前稀有度下的最大后缀数"""
        return self._max_suffix_for_rarity(self.rarity)

    # ---------- 环境重置与状态 ----------
    def reset(self):
        """重置环境到初始状态，并随机生成本回合的目标词缀数量"""
        total_target = self.equip['total_target']
        min_pre = self.equip.get('num_target_prefix_min', 0)
        max_pre = self.equip.get('num_target_prefix_max', self.max_prefix)
        min_suf = self.equip.get('num_target_suffix_min', 0)
        max_suf = self.equip.get('num_target_suffix_max', self.max_suffix)

        # 确保总目标数在合理范围内
        self.target_prefix_goal = random.randint(min_pre, min(max_pre, total_target))
        self.target_suffix_goal = total_target - self.target_prefix_goal
        if self.target_suffix_goal < min_suf:
            self.target_suffix_goal = min_suf
            self.target_prefix_goal = total_target - min_suf
        elif self.target_suffix_goal > max_suf:
            self.target_suffix_goal = max_suf
            self.target_prefix_goal = total_target - max_suf
        self.target_prefix_goal = max(min_pre, min(self.target_prefix_goal, max_pre))
        self.target_suffix_goal = total_target - self.target_prefix_goal

        self.rarity = 0                     # 0:普通, 1:魔法, 2:稀有
        self.prefixes = []                   # 实际拥有的前缀名称列表
        self.suffixes = []                    # 实际拥有的后缀名称列表
        self.target_prefix_count = 0         # 已获得的目标前缀数量
        self.target_suffix_count = 0          # 已获得的目标后缀数量

        return self._get_state()

    def _get_state(self):
        """将当前状态转换为神经网络输入向量（9维）"""
        rarity_onehot = [0, 0, 0]
        rarity_onehot[self.rarity] = 1
        state = rarity_onehot + [
            len(self.prefixes),
            self.target_prefix_count,
            len(self.suffixes),
            self.target_suffix_count,
            self.target_prefix_goal,
            self.target_suffix_goal
        ]
        return np.array(state, dtype=np.float32)

    # ---------- 动作合法性检查 ----------
    def _is_action_valid(self, action_idx: int) -> bool:
        """检查当前状态下动作是否合法"""
        action = self.actions[action_idx]
        cond = action.conditions
        effect = action.effect

        # 稀有度检查
        if 'rarity' in cond and self.rarity != cond['rarity']:
            return False
        if 'rarity_in' in cond and self.rarity not in cond['rarity_in']:
            return False

        etype = effect['type']
        if etype == 'add_random':
            # 确定动作执行后可能达到的稀有度（用于判断是否有空位）
            target_rarity = effect.get('upgrade_rarity_to', self.rarity)
            max_pre = self._max_prefix_for_rarity(target_rarity)
            max_suf = self._max_suffix_for_rarity(target_rarity)
            # 检查目标稀有度下是否有空位（即当前词缀数是否已满）
            if len(self.prefixes) >= max_pre and len(self.suffixes) >= max_suf:
                return False
            # 注意：即使当前稀有度下已满（如普通物品），只要目标稀有度有空位，也视为合法
        elif etype == 'remove_random':
            if len(self.prefixes) + len(self.suffixes) == 0:
                return False
        elif etype == 'reforge_one':
            if len(self.prefixes) + len(self.suffixes) == 0:
                return False
        # reroll_all 总是合法（只要稀有度满足）
        return True

    # ---------- 核心效果函数 ----------
    def _add_random_affix(self, effect_params: dict, target_rarity: int = None) -> bool:
        """
        添加一条随机词缀（可能指定前后缀池）
        :param effect_params: 效果参数字典，包含 'affix_pool' 等
        :param target_rarity: 动作执行后预期达到的稀有度（用于空位判断）
        :return: 是否成功添加
        """
        if target_rarity is None:
            target_rarity = self.rarity
        max_pre = self._max_prefix_for_rarity(target_rarity)
        max_suf = self._max_suffix_for_rarity(target_rarity)
        pool = effect_params.get('affix_pool', 'all')

        can_prefix = (pool in ['all', 'prefix']) and len(self.prefixes) < max_pre
        can_suffix = (pool in ['all', 'suffix']) and len(self.suffixes) < max_suf
        if not can_prefix and not can_suffix:
            return False

        if can_prefix and can_suffix:
            pos = random.choice(['prefix', 'suffix'])
        elif can_prefix:
            pos = 'prefix'
        else:
            pos = 'suffix'

        if pos == 'prefix':
            name = self.data.sample_prefix()
            self.prefixes.append(name)
            if name in self.target_prefix_pool:
                self.target_prefix_count += 1
        else:  # suffix
            name = self.data.sample_suffix()
            self.suffixes.append(name)
            if name in self.target_suffix_pool:
                self.target_suffix_count += 1
        return True

    def _remove_random_affix(self, effect_params: dict) -> bool:
        """随机移除一条词缀"""
        total = len(self.prefixes) + len(self.suffixes)
        if total == 0:
            return False
        r = random.randint(0, total - 1)
        if r < len(self.prefixes):
            removed = self.prefixes.pop(r)
            if removed in self.target_prefix_pool:
                self.target_prefix_count -= 1
        else:
            idx = r - len(self.prefixes)
            removed = self.suffixes.pop(idx)
            if removed in self.target_suffix_pool:
                self.target_suffix_count -= 1
        return True

    def _reforge_one_affix(self, effect_params: dict) -> bool:
        """随机选择一条词缀，将其重铸为新的随机词缀（替换）"""
        total = len(self.prefixes) + len(self.suffixes)
        if total == 0:
            return False
        r = random.randint(0, total - 1)
        if r < len(self.prefixes):
            old = self.prefixes.pop(r)
            if old in self.target_prefix_pool:
                self.target_prefix_count -= 1
            new = self.data.sample_prefix()
            self.prefixes.insert(r, new)
            if new in self.target_prefix_pool:
                self.target_prefix_count += 1
        else:
            idx = r - len(self.prefixes)
            old = self.suffixes.pop(idx)
            if old in self.target_suffix_pool:
                self.target_suffix_count -= 1
            new = self.data.sample_suffix()
            self.suffixes.insert(idx, new)
            if new in self.target_suffix_pool:
                self.target_suffix_count += 1
        return True

    def _reroll_all(self, effect_params: dict, target_rarity: int = None):
        """
        重置所有词缀，重新生成指定数量的词缀
        :param effect_params: 效果参数，包含 'num_affixes'
        :param target_rarity: 重置后预期达到的稀有度（用于确定最大词缀数）
        """
        if target_rarity is None:
            target_rarity = self.rarity
        # 清空当前词缀
        self.prefixes.clear()
        self.suffixes.clear()
        self.target_prefix_count = 0
        self.target_suffix_count = 0

        target_num = effect_params.get('num_affixes', 4)
        max_total = self.max_prefix + self.max_suffix  # 装备物理上限
        num = min(target_num, max_total)

        max_pre = self._max_prefix_for_rarity(target_rarity)
        max_suf = self._max_suffix_for_rarity(target_rarity)
        # 随机分配前后缀数量，但不超过目标稀有度下的上限
        pre = random.randint(0, min(max_pre, num))
        suf = num - pre
        if suf > max_suf:
            suf = max_suf
            pre = num - suf
            if pre > max_pre:
                pre = max_pre

        for _ in range(pre):
            self._add_random_affix({'affix_pool': 'prefix'}, target_rarity)
        for _ in range(suf):
            self._add_random_affix({'affix_pool': 'suffix'}, target_rarity)

    # ---------- 执行动作 ----------
    def step(self, action_idx: int):
        """
        执行一个动作，返回 (next_state, reward, done, info)
        """
        action = self.actions[action_idx]

        if not self._is_action_valid(action_idx):
            penalty_mult = self.reward_config.get('illegal_action_penalty_multiplier', 2)
            reward = -action.price * penalty_mult
            done = False
            return self._get_state(), reward, done, {"valid": False}

        effect = action.effect
        etype = effect['type']

        if etype == 'add_random':
            count = effect.get('count', 1)
            target_rarity = effect.get('upgrade_rarity_to', self.rarity)
            for _ in range(count):
                success = self._add_random_affix(effect, target_rarity)
                if not success:
                    break
            if 'upgrade_rarity_to' in effect:
                self.rarity = effect['upgrade_rarity_to']
        elif etype == 'remove_random':
            count = effect.get('count', 1)
            for _ in range(count):
                success = self._remove_random_affix(effect)
                if not success:
                    break
        elif etype == 'reforge_one':
            self._reforge_one_affix(effect)
        elif etype == 'reroll_all':
            target_rarity = effect.get('set_rarity', self.rarity)
            self._reroll_all(effect, target_rarity)
            if 'set_rarity' in effect:
                self.rarity = effect['set_rarity']
        else:
            raise ValueError(f"Unknown effect type: {etype}")

        reward = -action.price
        done = (self.target_prefix_count == self.target_prefix_goal and
                self.target_suffix_count == self.target_suffix_goal)
        if done:
            reward += self.reward_config.get('success_bonus', 0)

        return self._get_state(), reward, done, {"valid": True}