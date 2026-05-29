# tools/verify_environment.py
import sys
import os

# ==============================================================================
# 🌟 第一步：在任何导入之前，强行用系统底层管道向屏幕打字，验证脚本到底走没走
# ==============================================================================
sys.stdout.write("==============================================\n")
sys.stdout.write("▶ 进程已捕获：验证器主引擎正在突破静态检查区...\n")
sys.stdout.flush()

# --- 物理路径强行校正 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.stdout.write(f"📂 锚定项目根目录: {PROJECT_ROOT}\n")
sys.stdout.flush()

# ==============================================================================
# 🌟 第二步：使用动态捕获块进行异常隔离导入，防止静态导入期静默崩溃
# ==============================================================================
try:
    sys.stdout.write("⚙️  正在尝试引入 utils 模块...\n")
    sys.stdout.flush()
    from utils import GameData
    
    sys.stdout.write("⚙️  正在尝试引入 env 模块...\n")
    sys.stdout.flush()
    from env import GameEnv
    
    sys.stdout.write("✅ 所有模块导入成功！\n")
    sys.stdout.flush()
except Exception as import_error:
    sys.stdout.write(f"\n❌ [导入崩溃] 发现模块由于语法或路径问题无法加载：\n")
    import traceback
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    sys.exit(1)

import random
import numpy as np

# 强制终端输出支持特殊颜色高亮
os.system("")

class Color:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def run_automated_fuzz_test(game_data: GameData, steps: int = 1000):
    print(f"\n{Color.BOLD}{Color.BLUE}[1/2] 正在启动高频变异压力测试 (Fuzzing Test)... 共 {steps} 步{Color.END}", flush=True)
    extreme_equip_config = {
        "target_prefixes": ["不存在的词缀_A"], 
        "target_suffixes": ["不存在的词缀_B"],
        "max_prefix": 3,
        "max_suffix": 3
    }
    env = GameEnv(game_data, extreme_equip_config, {"success_bonus": 1000})
    env.reset()
    
    crash_count = 0
    illegal_execution_count = 0
    
    for i in range(steps):
        mask = env.get_valid_actions()
        valid_indices = np.where(mask == 1.0)[0]
        
        if len(valid_indices) == 0:
            env.reset()
            continue
            
        action_idx = random.choice(valid_indices)
        
        try:
            state, reward, done, info = env.step(action_idx)
            if not info.get("valid", True):
                illegal_execution_count += 1
        except Exception as e:
            crash_count += 1
            print(f"{Color.RED}❌ 发现物理引擎内核崩溃! 动作 ID: {action_idx}, 报错异常: {str(e)}{Color.END}", flush=True)
            
    print(f"\n{Color.BOLD}{Color.GREEN}📊 压力测试完结报告:{Color.END}", flush=True)
    print(f"  - 模拟运行总步数: {steps} 步")
    print(f"  - 运行时异常崩溃: {crash_count}")
    print(f"  - 掩码拦截失效漏洞: {illegal_execution_count}")
    return crash_count == 0 and illegal_execution_count == 0

def run_interactive_sandbox(game_data: GameData):
    print(f"\n{Color.BOLD}{Color.BLUE}[2/2] 正在切换至交互式人工逻辑验证沙盒...{Color.END}", flush=True)
    equip_config = {
        "target_prefixes": ["生命", "抗性"],
        "target_suffixes": [],
        "max_prefix": 3,
        "max_suffix": 3
    }
    env = GameEnv(game_data, equip_config, {"success_bonus": 500})
    env.reset()
    step_count = 0
    accumulated_reward = 0.0
    
    print(f"\n{Color.GREEN}🚀 沙盒初始化完毕！初始装备快照如下：{Color.END}", flush=True)
    print(env.render(), flush=True)
    
    while True:
        mask = env.get_valid_actions()
        print(f"\n{Color.BOLD}💡 当前物理层合法可点通货看板：{Color.END}", flush=True)
        
        for act in env.actions:
            status_text = f"{Color.GREEN}✅ 可点按{Color.END}" if mask[act.id] == 1.0 else f"{Color.RED}❌ 禁用{Color.END}"
            print(f"  [{act.id}] {act.name:<12} (基础价格: {act.price:<4} | 状态: {status_text})")
            
        print("-" * 60, flush=True)
        u_input = input(f"👉 请输入通货数字 {Color.YELLOW}[0-8]{Color.END} 点一下测试 (输入 {Color.RED}q{Color.END} 结束调试): ").strip()
        
        if u_input.lower() == 'q':
            print(f"\n{Color.YELLOW}已退出调试沙盒。环境完美卸载。{Color.END}", flush=True)
            break
            
        if not u_input.isdigit() or int(u_input) not in range(9):
            print(f"{Color.RED}⚠️ 错误：不合法的代号！请输入 0 到 8 之间的离散整数。{Color.END}", flush=True)
            continue
            
        act_id = int(u_input)
        
        # --- 针对人工体验的视觉拦截（直接封死输入） ---
        if mask[act_id] == 0.0:
            print(f"{Color.RED}⛔ [拒绝执行] 动作为非法操作！当前装备状态下，该通货已被物理规则完全封死，禁止点按！{Color.END}")
            continue # 直接跳过，不触发 env.step，不扣分，卡在原地
            
        next_state, reward, done, info = env.step(act_id)
        step_count += 1
        accumulated_reward += reward
        
        print(f"\n{Color.BOLD}🎬 【第 {step_count} 步执行回执】使用了通货: [{info.get('action_name', '未知')}]{Color.END}", flush=True)
        print(f"  - 即时 Reward 损耗: {reward:.1f} | 累计总 Reward 收益: {accumulated_reward:.1f}")
        print(f"  - 最新装备属性快照：\n{env.render()}", flush=True)

if __name__ == "__main__":
    ITEMS_PATH = os.path.join(PROJECT_ROOT, "config", "items_simple_currency.json")
    AFFIXES_PATH = os.path.join(PROJECT_ROOT, "config", "affixes.json")
    
    if not os.path.exists(ITEMS_PATH) or not os.path.exists(AFFIXES_PATH):
        sys.stdout.write(f"❌ 找不到 JSON 配置文件：\n  {ITEMS_PATH}\n")
        sys.stdout.flush()
        sys.exit(1)
        
    g_data = GameData(ITEMS_PATH, AFFIXES_PATH)
    run_automated_fuzz_test(g_data, steps=1000)
    run_interactive_sandbox(g_data)