[English](README.en.md) | 简体中文

(`FourierN1` 部份仍处于开发中，尚未完成）

# Wiki-GRx-Mujoco

本仓库提供了基于 Mujoco 的 Fourier N1 机器人的 RL 策略验证和可视化的代码实现。

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
   conda create -n wiki-grx-mujoco python=3.8 -y
   conda activate wiki-grx-mujoco
   ```

3. 依赖库安装
   ```
   pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
   ```

### Install Mujoco and Mujoco-viewer:

    ```
    pip install mujoco mujoco-python-viewer
    ```
    - <https://mujoco.org/>
    - <https://github.com/google-deepmind/mujoco/releases>

2. Load the models in Mujoco:

   get into the file location:
   ```
   ./mujoco-3.1.5/bin/
   ```
   and run:
   ```
   ./simulate
   ```
   and drag the `.xml` file that you want to view in robots folder

### Load trained policies in Mujoco:

1. get into the file location
     ```bash
     ./run/scripts
     ```

2. run the code with proper argument
     ```bash
     ./mjsim.py <robot_name> --load_model <path_to_model>
     ```

   **exmple:**

   load stand policy to control the robot GR1T1 to stand:

     ```bash
     ./mjsim.py gr1t1 --load_model /home/username/.../policy/stand_model_jit.pt
     ```

   or load the walk policy to control the robot GR1T2 to walk:

     ```bash
     ./mjsim.py gr1t2 --load_model /home/username/.../policy/walk_model_jit.pt
     ```

   You can modify the model parameters in `gr1tx_lower_limb.xml` and `robot_config`.


3. Control the robot by keyboard:

   After simulation started, you can press `.` to let the robot stand and press `/` to let the robot walk!

#

Thank you for your interest in the Fourier Intelligence GRx Robot Repositories.
We hope you find this resource helpful in your robotics projects!
