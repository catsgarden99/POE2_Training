# POE2 Crafting RL — 待办与痛点

## 高优先级

### [ ] 精华动作重构
- **现状**: 27 个独立精华动作 (action space 42)
- **目标**: 1 个通用精华动作 + 内部目标匹配
- **要点**:
  - 内置: 工艺词缀上限追踪 (多条工艺/符文)
  - 成本: 最低价精华作为固定基准 (暂不启用)
  - 自动匹配: 最优精华推进 target 词缀 (非目标不参与, 有争议, 留改)
  - 状态加 1 维: `cheapest_matching_essence_price`
- **阻塞**: 无

### [ ] 状态空间扩展
- **现状**: 13 维 (9 原始 + 1 亵渎 + 3 预兆)
- **痛点**: 无法区分好词缀和垃圾词缀; 缺少 ilvl、品质、装备类型、剩余预算
- **目标**: ~20-25 维 (初步方案)
  - 已有品质? 品质%\上限
  - 装备类型 one-hot (ring/amulet/belt/boots/gloves/helmet/chest/shield)
  - 已有词缀数量的目标匹配度编码
  - 剩余 budget
- **阻塞**: 需确认状态维度上限和 DQN 适配

### [ ] 奖励函数重构
- **现状**: `success_bonus` + `-cost` 二元 reward
- **痛点**: 混沌/剥离等随机操作无中间信号; 精华/崇高有确定性但也没部分奖励
- **目标**: 部分进度奖励 + 词缀价值评估
  - 每条词缀对比 target 给予部分 reward
  - 保留有用词缀给奖励 / 覆盖了好词缀给惩罚
  - 过早 stop 的惩罚
- **阻塞**: 待状态扩展后一起做

## 中优先级

### [ ] 终止动作 + 装备价值评估
- **现状**: `goal_reached` 硬触发 done
- **目标**: 1 个 `stop_crafting` 动作 + 装备价值 reward baseline
- **阻塞**: 需 reward 重构完成

### [ ] 亵渎系统完整实现
- **现状**: 只有 `has_desecrated` flag
- **缺失**: 亵渎池权重、多亵渎词缀交互、Light omen + 剥离链式操作
  - 实际: 装备可以有多个亵渎词缀
  - Light omen: 下次剥离只能移除亵渎词缀
  - 然后可以再次剥离 (不再受 omen 限制)
- **阻塞**: 低, 但依赖动作空间确定

## 低优先级

### [ ] 多装备类型支持
- **现状**: 只有 ring.json
- **目标**: helmet/boots/body/gloves/belt/shield/weapon 各一个 affix 数据文件
- **阻塞**: 数据量大, 等核心 RL 架构稳定后做

### [ ] 训练验证 + 超参数调优
- **现状**: 未跑过完整训练
- **痛点**: 不确定 DQN 能否收敛; 13 维 state + 256 hidden 可能 overparameterized
- **目标**: random agent baseline + DQN 训练曲线 + 超参网格搜索
- **阻塞**: 需以上架构调整完成后再跑

## 已知 Bug / 缺失功能

- [ ] Desecrated 词缀的 dual-group 支持 (如 `Strength|Intelligence` 双标签)
- [ ] cast_speed weight=1 (poe2db 数据如此, 社区无法验证)
