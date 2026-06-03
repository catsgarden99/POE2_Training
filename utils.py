# utils.py

from dataclasses import dataclass, field
import json
import random
from typing import List, Optional, Dict, Any


@dataclass
class Affix:
    name: str        # 词缀具体名称，例如 "T1_Maximum_Life"
    group: str       # 互斥组名称，例如 "Life"
    weight: int      # 抽样权重
    is_prefix: bool  # True为前缀，False为后缀

    def __repr__(self) -> str:
        return f"Affix(name={self.name}, group={self.group}, weight={self.weight}, prefix={self.is_prefix})"


@dataclass
class ItemAction:
    id: Any                             # 兼容整型或字符串的动作ID
    name: str                           # 动作名称，例如 "点金石"
    price: int                          # 通货价值/做装成本
    conditions: Dict[str, Any] = field(default_factory=dict)  # 做装硬性条件限制
    effect: Dict[str, Any] = field(default_factory=dict)      # 动作产生的游戏效果

    def __repr__(self) -> str:
        return f"ItemAction(id={self.id}, name={self.name}, price={self.price})"


class GameData:
    """游戏数据管理器：负责加载词缀池和通货定义，并提供严密的游戏规则抽样逻辑"""

    def __init__(self, items_path: str, affixes_path: str):
        """
        初始化游戏数据加载：完美适配声明式条件的精简通货文件
        """
        self.actions: List[ItemAction] = []
        self.prefixes: List[Affix] = []
        self.suffixes: List[Affix] = []
        self.affix_file: str = affixes_path

        # 1. 加载通货字典 (键为英文ID，如 Orb_of_Transmutation)
        with open(items_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)
            
        # 解析通货并塞入 action 空间
        action_id = 0  # 独立计数器，确保ID连续无缝
        for currency_key, data in raw_items.items():
            # 跳过被屏蔽的通货（enabled=false 或未声明enabled字段默认为true）
            if not data.get("enabled", True):
                continue

            action = ItemAction(
                id=action_id,  #  使用连续递增的离散数字作为 ID
                name=data["name"],
                price=data.get("price", 1), # 如果无价格默认1
                conditions=data.get("conditions", {}),  # 声明式字典
                effect=data.get("effect", {})           # 物理效果字典
            )
            self.actions.append(action)
            action_id += 1

        # 2. 加载词缀库 (兼容旧列表格式和新 per-equipment-type dict 格式)
        with open(affixes_path, "r", encoding="utf-8") as f:
            raw_affixes = json.load(f)
            if isinstance(raw_affixes, list):
                entries = raw_affixes
            elif isinstance(raw_affixes, dict):
                entries = raw_affixes.get("prefixes", []) + raw_affixes.get("suffixes", [])
            else:
                entries = []
            for entry in entries:
                affix = Affix(
                    name=entry.get("id", entry.get("name", "unknown")),
                    group=entry.get("group", "Unknown"),
                    weight=sum(t.get("weight", 1000) for t in entry.get("tiers", [{"weight": 1000}])),
                    is_prefix=entry.get("is_prefix", False),
                )
                if affix.is_prefix:
                    self.prefixes.append(affix)
                else:
                    self.suffixes.append(affix)

    def _load_items(self, path: str) -> None:
        """从 JSON 配置文件加载游戏动作/通货"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            self.actions.append(
                ItemAction(
                    id=item.get("id"),
                    name=item.get("name"),
                    price=item.get("price", 1),
                    conditions=item.get("conditions", {}),
                    effect=item.get("effect", {}),
                )
            )

    def _load_affixes(self, path: str) -> None:
        """
        从 JSON 配置文件加载全体词缀数据。
        支持两种格式：
        1. 拍平的传统列表结构: [{"name":..., "group":..., "weight":..., "is_prefix":...}, ...]
        2. 分层嵌套的旧结构: {"prefix": [{"name":..., "weight":...}], "suffix": [...]}
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 兼容处理：如果是旧版包含 'prefix' / 'suffix' 键的字典结构
        if isinstance(data, dict) and ("prefix" in data or "suffix" in data):
            for entry in data.get("prefix", []):
                # 旧数据没有 group，默认将名字本身作为 group 实现向下兼容
                affix = Affix(
                    name=entry["name"],
                    group=entry.get("group", entry["name"].split("_")[-1]),
                    weight=entry["weight"],
                    is_prefix=True,
                )
                self.prefixes.append(affix)
            for entry in data.get("suffix", []):
                affix = Affix(
                    name=entry["name"],
                    group=entry.get("group", entry["name"].split("_")[-1]),
                    weight=entry["weight"],
                    is_prefix=False,
                )
                self.suffixes.append(affix)
        # 新的标准扁平化列表结构
        elif isinstance(data, list):
            for entry in data:
                affix = Affix(
                    name=entry["name"],
                    group=entry["group"],
                    weight=entry["weight"],
                    is_prefix=entry["is_prefix"],
                )
                if affix.is_prefix:
                    self.prefixes.append(affix)
                else:
                    self.suffixes.append(affix)

    def sample_affix(
        self, current_affixes: List[Affix], roll_prefix: bool
    ) -> Optional[Affix]:
        """
        核心物理防冲突抽样机制：
        动态检测已有词缀的 group（词缀组），强制阻挡同组冲突词缀，按权重分配剩余池子概率进行点爆抽样。
        """
        pool = self.prefixes if roll_prefix else self.suffixes
        if not pool:
            return None

        # 1. 提取当前装备上所有的词缀互斥组标签
        existing_groups = {affix.group for affix in current_affixes}

        # 2. 动态过滤：剥离掉所有与当前装备词缀属于同一 Mod Group 的候选词缀
        valid_affixes = [a for a in pool if a.group not in existing_groups]

        if not valid_affixes:
            return None

        # 3. 计算相对概率分布并进行轮盘赌抽样
        weights = [a.weight for a in valid_affixes]
        chosen = random.choices(valid_affixes, weights=weights, k=1)[0]
        return chosen


# --- 单元基准测试单元 ---
if __name__ == "__main__":
    try:
        gd = GameData("config/items.json", "config/affixes.json")
        print(f"成功加载！前缀池大小: {len(gd.prefixes)} | 后缀池大小: {len(gd.suffixes)} | 动作数量: {len(gd.actions)}")
    except FileNotFoundError:
        print("未检测到正式路径文件，切换至就地初始化模式。")