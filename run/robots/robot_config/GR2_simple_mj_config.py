import numpy as np

class GR1T2SimpleCfg():
    class env:
        num_pri_obs = 168  
        num_actions = 21
        num_single_obs = 72
        frame_stack = 6
        num_observations = 432

    class control:
        action_scale = 0.25
        # PD Drive parameters:
        stiffness = {
            'hip_roll': 251.625, 'hip_yaw': 362.52, 'hip_pitch': 200,
            'knee_pitch': 200,
            'ankle_pitch': 10.98, 'ankle_roll': 10.98,
            'waist_yaw': 251.625, 'waist_pitch': 251.625, 'waist_roll': 251.625,
            'head_yaw': 112.06, 'head_pitch': 112.06, 'head_roll': 112.06,
            'shoulder_pitch': 92.85, 'shoulder_roll': 92.85, 'shoulder_yaw': 112.06,
            'elbow_pitch': 112.06,
            'wrist_yaw': 112.06, 'wrist_roll': 10.0, 'wrist_pitch': 10.0
        }  # [N*m/rad]
        damping = {
            'hip_roll': 14.72, 'hip_yaw': 10.0833, 'hip_pitch': 11,
            'knee_pitch': 11,
            'ankle_pitch': 0.6, 'ankle_roll': 0.6,
            'waist_yaw': 14.72, 'waist_pitch': 14.72, 'waist_roll': 14.72,
            'head_yaw': 3.1, 'head_pitch': 3.1, 'head_roll': 3.1,
            'shoulder_pitch': 2.575, 'shoulder_roll': 2.575, 'shoulder_yaw': 3.1,
            'elbow_pitch': 3.1,
            'wrist_yaw': 3.1, 'wrist_roll': 1.0, 'wrist_pitch': 1.0
        }



    class RobotConfig:

        kps = np.array([251.625, 362.52, 200, 200, 10.98, 10.98,
                        251.625, 362.52, 200, 200, 10.98, 10.98,
                        251.625,
                        92.85, 92.85, 112.06, 112.06,
                        92.85, 92.85, 112.06, 112.06,], dtype=np.double)  
        kds = np.array([14.72, 10.0833, 11, 11, 0.6, 0.6,
                        14.72, 10.0833, 11, 11, 0.6, 0.6,
                        14.72,
                        2.575, 2.575, 3.1, 3.1,
                        2.575, 2.575, 3.1, 3.1], dtype=np.double)
        tau_limit = np.array([60, 45, 130, 130, 8, 8, 
                              60, 45, 130, 130, 8, 8,
                              60,
                              30, 30, 30, 30,
                              30, 30, 30, 30], dtype=np.double)
        joint_nums = 21

    class MujocoModelPath:
        def __init__(self, path='./'):
            self.path = path


    class normalization:
        actions_max = np.array([
            0.79, 0.7, 0.7, 1.92, 0.52, 0.44,  # left leg
            0.09, 0.7, 0.7, 1.92, 0.52, 0.44,  # right leg
            0., # 1.05, # waist yaw
            1.92, 3.27, 2.97, 2.27, # left_arm
            1.92, 0.57, 2.97, 2.27 # right_arm
        ])
        actions_min = np.array([
            -0.09, -0.7, -1.75, -0.09, -1.05, -0.44, # left leg
            -0.79, -0.7, -1.75, -0.09, -1.05, -0.44, # right leg
            0., # -1.05, # waist_yaw
            -2.79, -0.57, -2.97, -2.27, # left_arm
            -2.79, -3.27, -2.97, -2.27 # right_arm
        ])

        clip_observations = 100.0
        clip_actions_max = actions_max + 60 / 180 * np.pi / 3
        clip_actions_min = actions_min - 60 / 180 * np.pi / 3
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
            'l_ankle_roll': 0.0,

            # right leg
            'r_hip_roll': 0.0,
            'r_hip_yaw': 0.,
            'r_hip_pitch': -0.2618,
            'r_knee_pitch': 0.5236,
            'r_ankle_pitch': -0.2618,
            'r_ankle_roll': 0.0,

            # waist
            'joint_waist_yaw': 0.0,
            # 'joint_waist_pitch': 0.0,
            # 'joint_waist_roll': 0.0,

            # head
            # 'joint_head_yaw': 0.0,
            # 'joint_head_pitch': 0.0,
            # 'joint_head_roll': 0.0,

            # left arm
            'l_shoulder_pitch': 0.0,
            'l_shoulder_roll': 0.0,
            'l_shoulder_yaw': 0.0,
            'l_elbow_pitch': -0.3,
            # 'l_wrist_yaw': 0.0,
            # 'l_wrist_roll': 0.0,
            # 'l_wrist_pitch': 0.0,

            # right_arm
            'r_shoulder_pitch': 0.0,
            'r_shoulder_roll': 0.0,
            'r_shoulder_yaw': 0.0,
            'r_elbow_pitch': -0.3,
            # 'r_wrist_yaw': 0.0,
            # 'r_wrist_roll': 0.0,
            # 'r_wrist_pitch': 0.0,
        }