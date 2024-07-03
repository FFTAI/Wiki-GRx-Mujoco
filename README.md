# Wiki-GRx-Mujoco

## Description
This repository provides an environment used to test the RL policy trained in NVIDIA's Isaac Gym on the GRx robot model in Mujoco.

## User Guide

1. Install Mujoco and Mujoco-viewer:

    ```
    pip install mujoco mujoco-viewer
    ```
    - <https://mujoco.org/>
    - <https://github.com/google-deepmind/mujoco/releases>
  
2. Install more dependencies:

    ```
    pip install -e .
    ```

3. Load the models in Mujoco:
   
   get into the file location:
   ```
   ./mujoco-3.1.5/bin/
   ```
   and run:
   ```
   ./simulate
   ```
   and drag the .xml file that you want to view in robots folder


Thank you for your interest in the Fourier Intelligence GRx Robot Repositories.
We hope you find this resource helpful in your robotics projects!
