# Wiki-GRx-Mujoco

## Description
This repository provides an environment used to test the RL policy trained in NVIDIA's Isaac Gym on the GRx robot model in Mujoco.

## User Guide
1. Create a conda environment:
   
   Create conda environment:
   ```
   conda create -n wiki-grx-mujoco python==3.8
   ```
   Activate the created environment:
   ```
   conda activate wiki-grx-mujoco
   ```

2. Install Mujoco and Mujoco-viewer:

    ```
    pip install mujoco mujoco-viewer
    ```
    - <https://mujoco.org/>
    - <https://github.com/google-deepmind/mujoco/releases>
  
3. Install more dependencies:

    ```
    pip install -e .
    ```

4. Load the models in Mujoco:
   
   get into the file location:
   ```
   ./mujoco-3.1.5/bin/
   ```
   and run:
   ```
   ./simulate
   ```
   and drag the `.xml` file that you want to view in robots folder

5. Load trained policies in Mujoco:
   
   get into the file location
   ```
   ./run/scripts
   ```
   then run:
   ```
   ./mjsim.py --load_model /home/username/.../policy/xxx_model_jit.pt
   ```
    You can modify the model parameters in `gr1t1_lower_limb.xml` and `robot_config`.



Thank you for your interest in the Fourier Intelligence GRx Robot Repositories.
We hope you find this resource helpful in your robotics projects!
