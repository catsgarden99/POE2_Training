# app.py

import streamlit as st
import pandas as pd
import json
import os
import subprocess
import torch # 新增
from utils import GameData
from env import GameEnv # 新增
from dqn import DQNAgent # 新增
from evaluate import generate_optimal_route # 新增

st.set_page_config(page_title="POE2 做装强化学习控制台", layout="wide")

st.title("🛠️ 《流放之路2》做装最低成本强化学习训练舱")
st.caption("基于 Double DQN + Action Masking 动态机制")

# 确保配置目录存在
os.makedirs("config", exist_ok=True)

# ----------------- 1. 数据加载 -----------------
@st.cache_data(ttl=2)  # 缓存2秒，方便刷新
def load_raw_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

try:
    game_data = GameData("config/items.json", "config/affixes.json")
    items_raw = load_raw_json("config/items.json")
except Exception as e:
    st.error(f"⚠️ 数据加载失败，请确保 config/ 目录下有 items.json 和 affixes.json。错误: {e}")
    st.stop()

# ----------------- 2. 侧边栏：通货资产与市场价维护 -----------------
st.sidebar.header("💰 通货市场价格/条件动态调整")
st.sidebar.info("在这里修改的价格会实时写入 items.json，强化学习会将价格作为负奖励（成本）进行寻优。")

# 将原始 json 转换为 Dataframe 供用户直观修改
df_items = pd.DataFrame(items_raw)
# 在侧边栏提供一个可交互的编辑表格
edited_df = st.sidebar.data_editor(
    df_items[["id", "name", "price"]], 
    hide_index=True,
    disabled=["id", "name"],
    column_config={"price": st.column_config.NumberColumn("当前价格 (代币/混沌)", min_value=1)}
)

# 保存价格改动
if st.sidebar.button("💾 保存通货价格改动"):
    for idx, row in edited_df.iterrows():
        items_raw[idx]["price"] = int(row["price"])
    with open("config/items.json", "w", encoding="utf-8") as f:
        json.dump(items_raw, f, indent=4, ensure_ascii=False)
    st.sidebar.success("🔥 价格已成功固化到 JSON，下次训练生效！")

# ----------------- 3. 主界面：做装成品目标配置 -----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 选择你本次想要追求的前缀 (Prefix)")
    prefix_names = [a.name for a in game_data.prefixes]
    target_prefixes = st.multiselect("点击挑选完美的理想前缀属性：", prefix_names)
    max_prefix = st.number_input("允许的最大前缀数量限制：", min_value=1, max_value=3, value=3)

with col2:
    st.subheader("🎯 选择你本次想要追求的后缀 (Suffix)")
    suffix_names = [a.name for a in game_data.suffixes]
    target_suffixes = st.multiselect("点击挑选完美的理想后缀属性：", suffix_names)
    max_suffix = st.number_input("允许的最大后缀数量限制：", min_value=1, max_value=3, value=3)

# ----------------- 4. 训练超参数微调 -----------------
st.subheader("🧠 强化学习策略微调")
with st.expander("点击展开：调整 AI 的智商与训练时长"):
    c1, c2, c3 = st.columns(3)
    with c1:
        num_episodes = st.slider("训练迭代轮数 (Episodes)", 1000, 50000, 10000, step=1000)
        learning_rate = st.select_slider("学习率 (LR)", options=[1e-4, 5e-4, 1e-3, 3e-3], value=1e-3)
    with c2:
        batch_size = st.selectbox("批次大小 (Batch Size)", ["32", "64", "128", "256"], index=2)
        gamma = st.slider("远期折现率 (Gamma)", 0.90, 0.99, 0.99, step=0.01)
    with c3:
        success_bonus = st.number_input("达成神装奖励分 (Success Bonus)", value=1000.0)

# ----------------- 5. 一键触发特训舱 -----------------
st.markdown("---")
if st.button("🚀 启动 AI 强化学习：开始寻找最低成本做装路径", type="primary"):
    if not target_prefixes and not target_suffixes:
        st.warning("❌ 靓仔，你至少得勾选一个目标词缀吧！不然 AI 不知道要做什么装备。")
    else:
        # 动态更新 equipment.json
        equip_config = {
            "target_prefixes": target_prefixes,
            "target_suffixes": target_suffixes,
            "max_prefix": int(max_prefix),
            "max_suffix": int(max_suffix)
        }
        with open("config/equipment.json", "w", encoding="utf-8") as f:
            json.dump(equip_config, f, indent=4, ensure_ascii=False)
            
        # 动态更新 training.json 中的超参数
        try:
            with open("config/training.json", "r", encoding="utf-8") as f:
                train_json = json.load(f)
        except FileNotFoundError:
            train_json = {"training": {}, "agent": {}, "epsilon": {"initial": 1.0, "min": 0.05, "decay_per_episode": 0.9995}, "reward": {}}
            
        train_json["training"]["num_episodes"] = int(num_episodes)
        train_json["training"]["batch_size"] = int(batch_size)
        train_json["training"]["target_update_freq"] = 200
        train_json["training"]["max_steps_per_episode"] = 50
        train_json["training"]["replay_buffer_capacity"] = 50000
        train_json["agent"]["learning_rate"] = float(learning_rate)
        train_json["agent"]["gamma"] = float(gamma)
        train_json["agent"]["hidden_dim"] = 128
        train_json["reward"]["success_bonus"] = float(success_bonus)
        
        with open("config/training.json", "w", encoding="utf-8") as f:
            json.dump(train_json, f, indent=4, ensure_ascii=False)

        st.success("📝 配置写入完毕！正在后台强制切入 `train.py` 进行重构训练...")
        
        # 在网页前端利用子进程跑训练，并实时把终端日志抓出来显示
        with st.spinner("AI 正在疯狂点通货洗装备中，请稍候..."):
            process = subprocess.Popen(["python", "train.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # --- 新增进度条和动态提示组件 ---
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            log_area = st.empty()
            log_text = ""
            
            # 实时读取终端输出并动态解析进度
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    log_text += output
                    
                    # 1. 动态裁剪日志：只保留最后 10 行，防止网页撑爆
                    lines = log_text.splitlines()
                    log_area.code("\n".join(lines[-10:]))
                    
                    # 2. 精准捕捉 train.py 吐出的实时进度
                    for line in reversed(lines):
                        if "PROGRESS:Episode" in line and "/" in line:
                            try:
                                # 提取 "PROGRESS:Episode 120/10000" 中的数字部分
                                part = line.split("PROGRESS:Episode")[1].split("|")[0].strip()
                                current_ep, total_ep = map(int, part.split("/"))
                                
                                # 计算百分比并无缝更新进度条
                                progress_percent = float(current_ep / total_ep)
                                progress_bar.progress(min(1.0, progress_percent))
                                
                                # 实时更新漂亮的文字状态提示
                                status_text.markdown(f"**⚡ 打造进度：** 已经模拟了 `{current_ep}` / `{total_ep}` 件装备 | **当前进度：** `{progress_percent*100:.1f}%`")
                                break  # 找到最新的一条进度就足够了，跳出当前查线循环
                            except Exception:
                                pass
            
            rc = process.poll()
            if rc == 0:
                # 训练完成，进度条拉满
                progress_bar.progress(1.0)
                status_text.success("✨ AI 已经完全掌握该装备的做装逻辑！")
                st.balloons()
                st.success("🎉 强化学习训练成功！最低成本做装大脑 (dqn_model.pth) 已成功炼成！")
                
                # 顺便跑一下评估，把最佳路线渲染在网页上
                st.markdown("---")
                st.markdown("### 🏆 AI 为你量身定制的最省钱做装路线树")
                
                # 在前端直接无缝实例化环境与加载刚刚练好的大脑
                try:
                    eval_env = GameEnv(game_data, equip_config, train_json['reward'])
                    eval_agent = DQNAgent(
                        state_dim=eval_env.state_dim,
                        action_dim=eval_env.num_actions,
                        lr=train_json['agent']['learning_rate'],
                        gamma=train_json['agent']['gamma'],
                        hidden_dim=train_json['agent']['hidden_dim']
                    )
                    # 强行无损读取热腾腾刚出炉的权重
                    eval_agent.policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=eval_agent.device))
                    eval_agent.policy_net.eval()
                    
                    # 运行路径推演
                    route, total_cost = generate_optimal_route(eval_env, eval_agent)
                    
                    st.metric(label="📊 达成目标预期总成本", value=f"{total_cost} 点通货价值")
                    
                    if not route:
                        st.warning("⚠️ 提示：AI 没能生成有效路线，可能因为训练轮数太少或目标定得太严苛，AI 迷路了。")
                    else:
                        # 漂亮的卡片化折叠步骤树展示
                        for s in route:
                            with st.expander(f"【步骤 {s['step']}】: 使用了 [{s['action']}] ➔ 消耗成本: {s['cost']}"):
                                st.code(s['equipment_status'], language="text")
                except Exception as eval_err:
                    st.error(f"渲染路线树时发生内部冲突: {eval_err}")
            else:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ 糟糕，后台训练脚本报错跑飞了，请检查终端日志。")