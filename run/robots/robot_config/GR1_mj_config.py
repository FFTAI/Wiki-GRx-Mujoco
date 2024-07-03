import numpy as np



class GR1T1LowerLimbCfg():
    class env:
        #NVIDIA 4090 has 16384 CUDA cores
        num_envs = 1 #Adjust according to CUDA cores numbers
        
        num_pri_obs = 168  ##
        num_actions = 10
        num_single_obs = 39
        num_observations = 39

    class SimConfig:
        #Simulate updaterate 50Hz
        sim_duration = 50.0
        dt = 0.001
        decimation = 20
    
    class RobotConfig:

        kps = np.array([57, 43, 114, 114, 15.3, 
                            57, 43, 114, 114, 15.3], dtype=np.double)  ##114
        kds = np.array([5.7, 4.3, 11.4, 11.4, 1.53, 
                            5.7, 4.3, 11.4, 11.4, 1.53], dtype=np.double)
        tau_limit = np.array([60, 45, 130, 130, 16, 
                                60, 45, 130, 130, 16], dtype=np.double)
        joint_nums = 10

    class normalization:
        actions_max = np.array([
            0.79, 0.7, 0.7, 1.92, 0.52,  # left leg
            0.09, 0.7, 0.7, 1.92, 0.52,  # right leg
        ])
        actions_min = np.array([
            -0.09, -0.7, -1.75, -0.09, -1.05,  # left leg
            -0.79, -0.7, -1.75, -0.09, -1.05,  # right leg
        ])

        clip_observations = 100.0
        clip_actions_max = np.array([1.1391, 1.0491, 1.0491, 2.2691, 0.8691,
                                        0.4391, 1.0491, 1.0491, 2.2691, 0.8691])
        clip_actions_min = np.array([-0.4391, -1.0491, -2.0991, -0.4391, -1.3991,
                                        -1.1391, -1.0491, -2.0991, -0.4391, -1.3991])
        class obs_scales:
            action = 1.0
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 1.0
            height_measurements = 1.0

        

    class init_state():
        pos = [0.0, 0.0, 0.95]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # left leg
            'l_hip_roll': 0.0,
            'l_hip_yaw': 0.,
            'l_hip_pitch': -0.2618,
            'l_knee_pitch': 0.5236,
            'l_ankle_pitch': -0.2618,
            #'l_ankle_roll': 0.0,

            # right leg
            'r_hip_roll': 0.0,
            'r_hip_yaw': 0.,
            'r_hip_pitch': -0.2618,
            'r_knee_pitch': 0.5236,
            'r_ankle_pitch': -0.2618,
            #'r_ankle_roll': 0.0,
        }

    class MujocoModelPath:
        def __init__(self, path='./'):
            self.path = path

    class control():
        action_scale = 1.0
        # PD Drive parameters:
        stiffness = {
            'hip_roll': 57,
            'hip_yaw': 43,
            'hip_pitch': 114,
            'knee_pitch': 114,
            'ankle_pitch': 15.3,
        }  # [N*m/rad]
        damping = {
            'hip_roll': stiffness['hip_roll'] / 10,
            'hip_yaw': stiffness['hip_yaw'] / 10,
            'hip_pitch': stiffness['hip_pitch'] / 10,
            'knee_pitch': stiffness['knee_pitch'] / 10,
            'ankle_pitch': stiffness['ankle_pitch'] / 10,
        }
        
