[简体中文](README.md) | English

(`FourierN1` part is still under development and not yet completed. The code in this repository is not yet usable.)

# Wiki-GRx-Mujoco

This repository provides the code implementation for RL policy verification and visualization of the Fourier N1 robot based on Mujoco.

> [!Note]
>
> Due to version updates of Mujoco, there have been significant changes in its Python development approach. Therefore, it is impossible to ensure compatibility with all versions.  
> This repository is primarily developed and tested based on the latest Mujoco 3.x.x version.

### Related Resources

* Mujoco: https://mujoco.org/  
* PyTorch: https://pytorch.org/  

### Installation Guide

1. Install Ubuntu 20.04 / Ubuntu 22.04 system.

2. Conda Environment Setup  
   ```
   # Install Miniconda  
   cd ~/Downloads  
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
   bash Miniconda3-latest-Linux-x86_64.sh  

   # Create a training environment  
   conda create -n wiki-grx-mujoco python=3.11 -y  
   conda activate wiki-grx-mujoco  
   ```

3. Dependency Installation  
   ```
   pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple  
   ```

### Usage Instructions

1. Launch Simulation  
   ```
   python run/scripts/mjsim.py --robot=gr1t1 --policy=walk_model_jit.pt  
   ```

---

Thank you for your interest in the Fourier Intelligence N1 robot project!  
We hope this resource will provide strong support for your robotics development!