import numpy as np
import mujoco
import mujoco_viewer
from tqdm import tqdm
from run.robots.robot_config.GR1_mj_config import GR1T1LowerLimbCfg
from run.robots.robot_config.GR2_mj_config import GR1T2LowerLimbCfg
from run.robots.robot_config.GR2_simple_mj_config import GR1T2SimpleCfg
import torch
import argparse
from pynput import keyboard

# Define the cmd class
class cmd:
    vx = 0.2
    vy = 0
    dyaw = 0

# Function to rotate quaternion inversely
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

# Function to get observation data
def get_obs(data):
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data.astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    return (q, dq, quat, omega)

# Function for PD control
def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

# Function to run the Mujoco simulation
def run_mujoco(policy, cfg):
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    count_lowlevel = 0
    gvec_tensor = torch.tensor([[0, 0, -1]], dtype=torch.float32)
    
    
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        '''
        q: position observation
        dq: speed observation
        quat: quaternion -> orientation observation, from imu
        omega: angular velocity observation, from imu
        '''

        q, dq, quat, omega = get_obs(data)
        # 28,27, 4, 3

        # for free flyer: q:7, dq:6
        q_base = q[0:3]
        dq_base = dq[0:3]
        quat = q[3:7]
        omega = dq[3:6]
        
        # for rest joint: 21
        q_joint = q[7:]
        dq_joint = dq[6:]

        joint_names = ['l_hip_roll',
                       'l_hip_yaw',
                       'l_hip_pitch',
                       'l_knee_pitch',
                       'l_ankle_pitch',
                       'l_ankle_roll',

                       'r_hip_roll',
                       'r_hip_yaw',
                       'r_hip_pitch',
                       'r_knee_pitch',
                       'r_ankle_pitch',
                       'r_ankle_roll',

                       'joint_waist_yaw',

                       'l_shoulder_pitch',
                       'l_shoulder_roll',
                       'l_shoulder_yaw',
                       'l_elbow_pitch',

                       'r_shoulder_pitch',
                       'r_shoulder_roll',
                       'r_shoulder_yaw',
                       'r_elbow_pitch']
        
        default_joint_angles = np.array([cfg.init_state.default_joint_angles[name] for name in joint_names])
        # default_joint_angles = default_joint_angles[-cfg.env.num_actions:]

        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32) # 75
            quat_tensor = torch.tensor([quat], dtype=torch.float32)
            quat_tensor = quat_tensor[:, [1, 2, 3, 0]]
            omega_tensor = torch.tensor([omega], dtype=torch.float32)
            quat_proj = quat_rotate_inverse(quat_tensor, gvec_tensor)
            omega_proj = quat_rotate_inverse(quat_tensor, omega_tensor)


            # import pdb 
            # pdb.set_trace()

            obs[0, 0] = cmd.vx
            obs[0, 1] = cmd.vy
            obs[0, 2] = cmd.dyaw

            obs[0, 3:6] = omega_proj # (base angular velocity)

            gravity = model.opt.gravity

            # Gravity components, x, y, z
            obs[0, 6] = gravity[0] 
            obs[0, 7] = gravity[1]
            obs[0, 8] = gravity[2]

            # position
            obs[0, 9] = q_joint[12] # joint_waist_yaw
            obs[0, 10:14] = q_joint[13:17] # left_arm
            obs[0, 14:18] = q_joint[17:21] # right_arm
            obs[0, 18:24] = q_joint[0:6] # left_leg
            obs[0, 24:30] = q_joint[6:12] # right_leg

            # velocity
            obs[0, 30] = dq_joint[12] # joint_waist_yaw
            obs[0, 31:35] = dq_joint[13:17] # left_arm
            obs[0, 35:39] = dq_joint[17:21] # right_arm
            obs[0, 39:45] = dq_joint[0:7] # left_leg
            obs[0, 45:51] = dq_joint[7:12] # right_leg


            # last_action
            obs[0, 51] = action[12] # joint_waist_yaw
            obs[0, 52:56] = action[13:17] # left_arm
            obs[0, 56:60] = action[17:21] # right_arm
            obs[0, 60:66] = action[0:7] # left_leg
            obs[0, 66:72] = action[7:12] # right_leg


            # base x,y,z
            obs[0, 72:75] = q_base



            # obs[0, 0:3] = omega_proj
            # obs[0, 3:6] = quat_proj
            # obs[0, 6] = cmd.vx
            # obs[0, 7] = cmd.vy
            # obs[0, 8] = cmd.dyaw
            
            # obs[0, 9:19] = (q - default_joint_angles) * cfg.normalization.obs_scales.dof_pos
            # obs[0, 19:29] = dq * cfg.normalization.obs_scales.dof_vel
            # obs[0, 29:39] = action

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            policy_input[0, :cfg.env.num_single_obs] = obs[0, :cfg.env.num_single_obs]

            action[:] = policy.forward(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, cfg.normalization.clip_actions_min, cfg.normalization.clip_actions_max)

            target_q = (action + default_joint_angles) * cfg.control.action_scale
            
        tau = (target_q - q) * cfg.RobotConfig.kps - dq * cfg.RobotConfig.kds
        # tau = np.clip(tau, -cfg.RobotConfig.tau_limit, cfg.RobotConfig.tau_limit)
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1
    viewer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('robot_id', type=str, help='Path to the model to load.')
    parser.add_argument('--load_model', type=str, required=True, help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()
    if args.robot_id == 'gr1t1':
        RobotConfig = GR1T1LowerLimbCfg
    elif args.robot_id == 'gr1t2':
        RobotConfig = GR1T2LowerLimbCfg
    elif args.robot_id =='gr1t2_simple':
        RobotConfig = GR1T2SimpleCfg

    class Sim2simCfg(RobotConfig):
        class sim_config:
            mujoco_model_path = f'../robots/{args.robot_id}/scene.xml'
            sim_duration = 70.0
            dt = 0.001
            decimation = 20
    
    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())



    '''
    flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
    flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0] # x command
    flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1] # y command
    flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2] # yaw command
    flipped_proprioceptive_obs[:, :, 3] = -proprioceptive_obs[:, :, 3] # base ang vel roll
    flipped_proprioceptive_obs[:, :, 4] =  proprioceptive_obs[:, :, 4] # base ang vel pitch
    flipped_proprioceptive_obs[:, :, 5] = -proprioceptive_obs[:, :, 5] # base ang vel yaw
    flipped_proprioceptive_obs[:, :, 6] =  proprioceptive_obs[:, :, 6] # projected gravity x
    flipped_proprioceptive_obs[:, :, 7] = -proprioceptive_obs[:, :, 7] # projected gravity y
    flipped_proprioceptive_obs[:, :, 8] =  proprioceptive_obs[:, :, 8] # projected gravity z

    # Joint Pos
    flipped_proprioceptive_obs[:, :,  9] = -proprioceptive_obs[:, :,  9] # joint_waist_yaw pos

    flipped_proprioceptive_obs[:, :, 10] =  proprioceptive_obs[:, :, 14] # joint_l_shoulder_pitch pos
    flipped_proprioceptive_obs[:, :, 11] = -proprioceptive_obs[:, :, 15] # joint_l_shoulder_roll pos
    flipped_proprioceptive_obs[:, :, 12] = -proprioceptive_obs[:, :, 16] # joint_l_shoulder_yaw pos
    flipped_proprioceptive_obs[:, :, 13] =  proprioceptive_obs[:, :, 17] # joint_l_elbow_pitch pos

    flipped_proprioceptive_obs[:, :, 14] =  proprioceptive_obs[:, :, 10] # joint_r_shoulder_pitch pos
    flipped_proprioceptive_obs[:, :, 15] = -proprioceptive_obs[:, :, 11] # joint_r_shoulder_roll pos
    flipped_proprioceptive_obs[:, :, 16] = -proprioceptive_obs[:, :, 12] # joint_r_shoulder_yaw pos
    flipped_proprioceptive_obs[:, :, 17] =  proprioceptive_obs[:, :, 13] # joint_r_elbow_pitch pos

    flipped_proprioceptive_obs[:, :, 18] = -proprioceptive_obs[:, :, 24] # joint_left_hip_roll pos
    flipped_proprioceptive_obs[:, :, 19] = -proprioceptive_obs[:, :, 25] # joint_left_hip_yaw pos
    flipped_proprioceptive_obs[:, :, 20] =  proprioceptive_obs[:, :, 26] # joint_left_hip_pitch pos
    flipped_proprioceptive_obs[:, :, 21] =  proprioceptive_obs[:, :, 27] # joint_left_knee_pitch pos
    flipped_proprioceptive_obs[:, :, 22] =  proprioceptive_obs[:, :, 28] # joint_left_ankle_pitch pos
    flipped_proprioceptive_obs[:, :, 23] = -proprioceptive_obs[:, :, 29] # joint_left_ankle_roll pos

    flipped_proprioceptive_obs[:, :, 24] = -proprioceptive_obs[:, :, 18] # joint_right_hip_roll pos
    flipped_proprioceptive_obs[:, :, 25] = -proprioceptive_obs[:, :, 19] # joint_right_hip_yaw pos
    flipped_proprioceptive_obs[:, :, 26] =  proprioceptive_obs[:, :, 20] # joint_right_hip_pitch pos
    flipped_proprioceptive_obs[:, :, 27] =  proprioceptive_obs[:, :, 21] # joint_right_knee_pitch pos
    flipped_proprioceptive_obs[:, :, 28] =  proprioceptive_obs[:, :, 22] # joint_right_ankle_pitch pos
    flipped_proprioceptive_obs[:, :, 29] = -proprioceptive_obs[:, :, 23] # joint_right_ankle_roll pos

    # Joint Vel
    flipped_proprioceptive_obs[:, :, 30] = -proprioceptive_obs[:, :, 30] # joint_waist_yaw vel

    flipped_proprioceptive_obs[:, :, 31] =  proprioceptive_obs[:, :, 35] # joint_l_shoulder_pitch vel
    flipped_proprioceptive_obs[:, :, 32] = -proprioceptive_obs[:, :, 36] # joint_l_shoulder_roll vel
    flipped_proprioceptive_obs[:, :, 33] = -proprioceptive_obs[:, :, 37] # joint_l_shoulder_yaw vel
    flipped_proprioceptive_obs[:, :, 34] =  proprioceptive_obs[:, :, 38] # joint_l_elbow_pitch vel

    flipped_proprioceptive_obs[:, :, 35] =  proprioceptive_obs[:, :, 31] # joint_r_shoulder_pitch vel
    flipped_proprioceptive_obs[:, :, 36] = -proprioceptive_obs[:, :, 32] # joint_r_shoulder_roll vel
    flipped_proprioceptive_obs[:, :, 37] = -proprioceptive_obs[:, :, 33] # joint_r_shoulder_yaw vel
    flipped_proprioceptive_obs[:, :, 38] =  proprioceptive_obs[:, :, 34] # joint_r_elbow_pitch vel

    flipped_proprioceptive_obs[:, :, 39] = -proprioceptive_obs[:, :, 45] # joint_left_hip_roll vel
    flipped_proprioceptive_obs[:, :, 40] = -proprioceptive_obs[:, :, 46] # joint_left_hip_yaw vel
    flipped_proprioceptive_obs[:, :, 41] =  proprioceptive_obs[:, :, 47] # joint_left_hip_pitch vel
    flipped_proprioceptive_obs[:, :, 42] =  proprioceptive_obs[:, :, 48] # joint_left_knee_pitch vel
    flipped_proprioceptive_obs[:, :, 43] =  proprioceptive_obs[:, :, 49] # joint_left_ankle_pitch vel
    flipped_proprioceptive_obs[:, :, 44] = -proprioceptive_obs[:, :, 50] # joint_left_ankle_roll vel

    flipped_proprioceptive_obs[:, :, 45] = -proprioceptive_obs[:, :, 39] # joint_right_hip_roll vel
    flipped_proprioceptive_obs[:, :, 46] = -proprioceptive_obs[:, :, 40] # joint_right_hip_yaw vel
    flipped_proprioceptive_obs[:, :, 47] =  proprioceptive_obs[:, :, 41] # joint_right_hip_pitch vel
    flipped_proprioceptive_obs[:, :, 48] =  proprioceptive_obs[:, :, 42] # joint_right_knee_pitch vel
    flipped_proprioceptive_obs[:, :, 49] =  proprioceptive_obs[:, :, 43] # joint_right_ankle_pitch vel
    flipped_proprioceptive_obs[:, :, 50] = -proprioceptive_obs[:, :, 44] # joint_right_ankle_roll vel

    # Last Action
    flipped_proprioceptive_obs[:, :, 51] = -proprioceptive_obs[:, :, 51] # joint_waist_yaw last action

    flipped_proprioceptive_obs[:, :, 52] =  proprioceptive_obs[:, :, 56] # joint_l_shoulder_pitch last action
    flipped_proprioceptive_obs[:, :, 53] = -proprioceptive_obs[:, :, 57] # joint_l_shoulder_roll last action
    flipped_proprioceptive_obs[:, :, 54] = -proprioceptive_obs[:, :, 58] # joint_l_shoulder_yaw last action
    flipped_proprioceptive_obs[:, :, 55] =  proprioceptive_obs[:, :, 59] # joint_l_elbow_pitch last action

    flipped_proprioceptive_obs[:, :, 56] =  proprioceptive_obs[:, :, 52] # joint_r_shoulder_pitch last action
    flipped_proprioceptive_obs[:, :, 57] = -proprioceptive_obs[:, :, 53] # joint_r_shoulder_roll last action
    flipped_proprioceptive_obs[:, :, 58] = -proprioceptive_obs[:, :, 54] # joint_r_shoulder_yaw last action
    flipped_proprioceptive_obs[:, :, 59] =  proprioceptive_obs[:, :, 55] # joint_r_elbow_pitch last action

    flipped_proprioceptive_obs[:, :, 60] = -proprioceptive_obs[:, :, 66] # joint_left_hip_roll last action
    flipped_proprioceptive_obs[:, :, 61] = -proprioceptive_obs[:, :, 67] # joint_left_hip_yaw last action
    flipped_proprioceptive_obs[:, :, 62] =  proprioceptive_obs[:, :, 68] # joint_left_hip_pitch last action
    flipped_proprioceptive_obs[:, :, 63] =  proprioceptive_obs[:, :, 69] # joint_left_knee_pitch last action
    flipped_proprioceptive_obs[:, :, 64] =  proprioceptive_obs[:, :, 70] # joint_left_ankle_pitch last action
    flipped_proprioceptive_obs[:, :, 65] = -proprioceptive_obs[:, :, 71] # joint_left_ankle_roll last action

    flipped_proprioceptive_obs[:, :, 66] = -proprioceptive_obs[:, :, 60] # joint_right_hip_roll last action
    flipped_proprioceptive_obs[:, :, 67] = -proprioceptive_obs[:, :, 61] # joint_right_hip_yaw last action
    flipped_proprioceptive_obs[:, :, 68] =  proprioceptive_obs[:, :, 62] # joint_right_hip_pitch last action
    flipped_proprioceptive_obs[:, :, 69] =  proprioceptive_obs[:, :, 63] # joint_right_knee_pitch last action
    flipped_proprioceptive_obs[:, :, 70] =  proprioceptive_obs[:, :, 64] # joint_right_ankle_pitch last action
    flipped_proprioceptive_obs[:, :, 71] = -proprioceptive_obs[:, :, 65] # joint_right_ankle_roll last action

    '''
