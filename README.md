# Wiki-GRx-Mujoco
<img src="./run/robots/gr1t1/gr1t1.png" width="300" height="360" />
<img src="./run/robots/gr1t2/gr1t2.png" width="300" height="360" />

## Description
This repository provides an environment used to test the RL policy trained in NVIDIA's Isaac Gym on the GRx robot model in Mujoco. Also you can use mujoco to review your robot.

## User Guide
### Create a conda environment:
   
   Create conda environment:
   ```
   conda create -n wiki-grx-mujoco python==3.8
   ```
   Activate the created environment:
   ```
   conda activate wiki-grx-mujoco
   ```

### Install Mujoco and Mujoco-viewer:

    ```
    pip install mujoco mujoco-python-viewer
    ```
    - <https://mujoco.org/>
    - <https://github.com/google-deepmind/mujoco/releases>
  
1. Install more dependencies:
    ```
    cd wiki-grx-mujoco
    ```

    ```
    pip install -e .
    ```

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
