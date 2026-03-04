# POE2_Training
流放2做装训练

## tips

```sh
# 创建名为 crafting_rl 的环境，指定 Python 3.9
conda create -n poe_crafting python=3.9 -y

# 激活环境
conda activate poe_crafting

# 安装 PyTorch（CPU 版，如需 GPU 请根据官网替换）
pip install torch torchvision torchaudio

# 安装 PyTorch（GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install numpy matplotlib tensorboard jupyter
```