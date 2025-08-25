import numpy


class N1Config:
    class sim:
        model_path = None
        sim_duration = 60.0
        dt = 0.001
        decimation = 20

    class env:
        num_dofs = 6 + 6 + 1 + 5 + 5
        num_obs = 48
        num_stack = 5
        num_stack_obs = num_obs * num_stack
        num_actions = 6 + 6 + 1

    class command:
        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel_yaw = 0

    class robot:
        kps = numpy.array([
            180.0, 120.0, 90.0, 120.0, 45.0, 45.0,  # left leg
            180.0, 120.0, 90.0, 120.0, 45.0, 45.0,  # right leg
            90.0,  # waist
            90.0, 45.0, 45.0, 45.0, 45.0,  # left arm
            90.0, 45.0, 45.0, 45.0, 45.0,  # right arm
        ], dtype=numpy.double)
        kds = numpy.array([
            10.0, 10.0, 8.0, 8.0, 2.5, 2.5,  # left leg
            10.0, 10.0, 8.0, 8.0, 2.5, 2.5,  # right leg
            8.0,  # waist
            8.0, 2.5, 2.5, 2.5, 2.5,  # left arm
            8.0, 2.5, 2.5, 2.5, 2.5,  # right arm
        ], dtype=numpy.double)
        tau_limit = numpy.array([
            95, 54, 54, 95, 30, 30,  # left leg
            95, 54, 54, 95, 30, 30,  # right leg
            54,  # waist
            54, 30, 30, 30, 30,  # left arm
            54, 30, 30, 30, 30,  # right arm
        ], dtype=numpy.double)

        joint_nums = 6 + 6 + 1

    class normalization:
        actions_max = numpy.array([
            2.618, 1.571, 1.571, 2.356, 0.436, 0.785,  # left leg
            2.618, 0.262, 1.571, 2.356, 0.436, 0.785,  # right leg
            2.618,  # waist
        ])
        actions_min = numpy.array([
            -2.618, -0.262, -1.571, -0.087, -0.436, -0.785,  # left leg
            -2.618, -1.571, -1.571, -0.087, -0.436, -0.785,  # right leg
            -2.618,  # waist
        ])

        clip_observations = 100.0

        clip_actions_max = \
            actions_max \
            + numpy.array([
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # left leg
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # right leg
                0.5,  # waist
            ])
        clip_actions_min = \
            actions_min \
            - numpy.array([
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # left leg
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # right leg
                0.5,  # waist
            ])

        class obs_scales:
            action = 1.00
            lin_vel = 1.00  # map 1.0 m/s -> 1.0
            ang_vel = 1.00  # map 1.0 rad/s -> 1.0
            gravity = 1.00
            dof_pos = 1.00
            dof_vel = 0.10  # map 6.28 rad/s -> 0.628
            height_measurements = 5.0  # map 0.2 m -> 1.0

    class init_state:
        pos = [0.0, 0.0, 0.70]  # x,y,z [m]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # left leg
            "left_hip_pitch_joint": -numpy.deg2rad(14.0),
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_pitch_joint": +numpy.deg2rad(29.5),
            "left_ankle_roll_joint": 0.0,
            "left_ankle_pitch_joint": -numpy.deg2rad(13.7),

            # right leg
            "right_hip_pitch_joint": -numpy.deg2rad(14.0),
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_pitch_joint": +numpy.deg2rad(29.5),
            "right_ankle_roll_joint": 0.0,
            "right_ankle_pitch_joint": -numpy.deg2rad(13.7),

            # waist
            "waist_yaw_joint": 0.0,
        }

    class observation:
        num_dofs = 6 + 6 + 1
        index_dofs = [
            0, 1, 2, 3, 4, 5,  # left leg
            6, 7, 8, 9, 10, 11,  # right leg
            12,  # waist
        ]
        num_actions = 6 + 6 + 1
        index_actions = [
            0, 1, 2, 3, 4, 5,  # left leg
            6, 7, 8, 9, 10, 11,  # right leg
            12,  # waist
        ]

    class control:
        action_names = [
            # leg
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee_pitch",
            "ankle_roll",
            "ankle_pitch",

            # waist
            "waist_yaw",
        ]
        action_scale = {
            # leg
            "hip_pitch": 1,
            "hip_roll": 1,
            "hip_yaw": 1,
            "knee_pitch": 1,
            "ankle_roll": 1,
            "ankle_pitch": 1,

            # waist
            "waist_yaw": 1,
        }

    class policy:
        path = "policy_jit_walk.pt"
