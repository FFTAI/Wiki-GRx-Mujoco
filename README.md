[English](README.en.md) | 简体中文

(`FourierN1` 部份仍处于开发中，尚未完成，本仓库代码尚不可用）

# Wiki-GRx-Mujoco

本仓库提供了基于 Mujoco 的 Fourier N1 机器人的 RL 策略验证和可视化的代码实现。

> [!NOTE]
> 由于 Mujoco 的版本更新，其 python 开发方式变更较多，因此，不可能做到所有版本都兼容。
> 本仓库主要基于最新的 Mujoco 3.x.x 版本进行开发和测试。

### 相关资源

* Mujoco: https://mujoco.org/
* pytorch: https://pytorch.org/

### 安装指南

1. 安装 Ubuntu 20.04 / Ubuntu 22.04 系统

2. Conda环境配置
   ```
   # 安装Miniconda
   cd ~/Downloads
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh

   # 创建训练环境
   conda create -n wiki-grx-mujoco python=3.11 -y
   conda activate wiki-grx-mujoco
   ```

3. 依赖库安装
   ```
   pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
   ```

### 使用说明

1. 启动仿真
   ```
   python main.py --robot=N1 --policy=policy_jit_walk.pt
   ```

---

感谢您对傅利叶智能 N1 机器人项目的关注！
希望本资源能为您的机器人开发提供有力支持！