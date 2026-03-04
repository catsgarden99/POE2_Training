import json
import random
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Affix:
    """词缀数据类，包含名称和权重"""
    name: str
    weight: int

@dataclass
class ItemAction:
    """道具数据类，包含ID、名称、价格、使用条件、效果"""
    id: str
    name: str
    price: int
    conditions: Dict[str, Any]
    effect: Dict[str, Any]

class GameData:
    """加载并存储游戏静态数据（道具、词缀）"""
    def __init__(self, items_file: str, affixes_file: str):
        # 加载道具配置
        with open(items_file, 'r', encoding='utf-8') as f:
            items_data = json.load(f)
        self.actions = [ItemAction(**item) for item in items_data]
        
        # 加载词缀配置
        with open(affixes_file, 'r', encoding='utf-8') as f:
            affixes_data = json.load(f)
        self.prefixes = [Affix(**a) for a in affixes_data['prefix']]
        self.suffixes = [Affix(**a) for a in affixes_data['suffix']]
        
        # 计算词缀总数（用于抽样）
        self.num_prefixes = len(self.prefixes)
        self.num_suffixes = len(self.suffixes)
        
        # 预计算累积权重
        self.prefix_cum_weights = []
        cum = 0
        for a in self.prefixes:
            cum += a.weight
            self.prefix_cum_weights.append(cum)
        self.suffix_cum_weights = []
        cum = 0
        for a in self.suffixes:
            cum += a.weight
            self.suffix_cum_weights.append(cum)

    def sample_prefix(self):
        total = self.prefix_cum_weights[-1]
        r = random.uniform(0, total)
        # 二分查找
        import bisect
        idx = bisect.bisect_left(self.prefix_cum_weights, r)
        return self.prefixes[idx].name

    def sample_suffix(self):
        total = self.suffix_cum_weights[-1]
        r = random.uniform(0, total)
        import bisect
        idx = bisect.bisect_left(self.suffix_cum_weights, r)
        return self.suffixes[idx].name