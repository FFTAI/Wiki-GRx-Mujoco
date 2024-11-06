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
from collections import deque
from scipy.spatial.transform import Rotation as R

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
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    gvec = r.apply(np.array([0, 0, -1]), inverse=True).astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    return (q, dq, quat, omega, gvec)

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
    target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    count_lowlevel = 0
    gvec_tensor = torch.tensor([[0, 0, -1]], dtype=torch.float32)
    init_state = True
    
    # hist_obs = deque()
    # for _ in range(cfg.env.frame_stack):
    #     hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
    policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)

    # joint_names = ['joint_waist_yaw',
                   
    #             'l_shoulder_pitch',
    #             'l_shoulder_roll',
    #             'l_shoulder_yaw',
    #             'l_elbow_pitch',

    #             'r_shoulder_pitch',
    #             'r_shoulder_roll',
    #             'r_shoulder_yaw',
    #             'r_elbow_pitch',
                
    #             'l_hip_roll',
    #             'l_hip_yaw',
    #             'l_hip_pitch',
    #             'l_knee_pitch',
    #             'l_ankle_pitch',
    #             'l_ankle_roll',

    #             'r_hip_roll',
    #             'r_hip_yaw',
    #             'r_hip_pitch',
    #             'r_knee_pitch',
    #             'r_ankle_pitch',
    #             'r_ankle_roll']
    joint_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee_pitch', 'l_ankle_pitch', 'l_ankle_roll',
                    'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee_pitch', 'r_ankle_pitch', 'r_ankle_roll',
                    'joint_waist_yaw',
                      'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow_pitch',
                    'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow_pitch']
    default_joint_angles = np.array([cfg.init_state.default_joint_angles[name] for name in joint_names])
    print(default_joint_angles.shape)


    gravity_tensor = torch.tensor([[0, 0, -1]],dtype=torch.float32)
                       
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        '''
        q: position observation
        dq: speed observation
        quat: quaternion -> orientation observation, from imu
        omega: angular velocity observation, from imu
        '''

        q, dq, quat, omega, gvec = get_obs(data)
        # 28,27, 4, 3

        # for free flyer: q:7, dq:6
        q_base = q[0:3]
        dq_base = dq[0:3]
        
        # for rest joint: 21
        q_joint = q[7:] - default_joint_angles
        dq_joint = dq[6:]
        mujoco_control_q = np.zeros_like(target_q)

        # if count_lowlevel < 200:
        #     target_q = default_joint_angles
        #     print("running init")
        


        if count_lowlevel % cfg.sim_config.decimation == 0:# and count_lowlevel >= 200:

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32) # 72

            quat_tensor = torch.tensor(np.array([quat]), dtype=torch.float32)
            omega_tensor = torch.tensor([omega], dtype=torch.float32)

            # # quat_proj = quat_rotate_inverse(quat_tensor, gvec_tensor)
            # omega_proj = quat_rotate_inverse(quat_tensor, omega_tensor)
            
            quat_proj = quat_rotate_inverse(quat_tensor, gravity_tensor)
            # omega_proj = quat_rotate_inverse(omega_tensor,gravity_tensor)


            # import pdb 
            # pdb.set_trace()

            obs[0, 0] = cmd.vx
            obs[0, 1] = cmd.vy
            obs[0, 2] = cmd.dyaw

            obs[0, 3:6] = np.array(omega) * 0.25 # (base angular velocity)
            # print("omega: ", obs[0, 3:6])

            # gravity = model.opt.gravity
            # Gravity components, x, y, z
            # obs[0, 6] = gvec[0]
            # obs[0, 7] = gvec[1]
            # obs[0, 8] = gvec[2]
            obs[0, 6:9] = np.array(quat_proj[0, :3]) # gravity vector

            # position
            obs[0, 9] = q_joint[12] # joint_waist_yaw
            obs[0, 10:14] = q_joint[13:17] # left_arm
            obs[0, 14:18] = q_joint[17:21] # right_arm
            obs[0, 18:24] = q_joint[0:6] # left_leg
            obs[0, 24:30] = q_joint[6:12] # right_leg
            # obs[0, 21] = q_joint[12] # joint_waist_yaw
            # obs[0, 22:26] = q_joint[13:17] # left_arm
            # obs[0, 26:30] = q_joint[17:21] # right_arm
            # obs[0, 9:15] = q_joint[0:6] # left_leg
            # obs[0, 15:21] = q_joint[6:12] # right_leg

            # velocity
            obs[0, 30] = dq_joint[12] # joint_waist_yaw
            obs[0, 31:35] = dq_joint[13:17] # left_arm
            obs[0, 35:39] = dq_joint[17:21] # right_arm
            obs[0, 39:45] = dq_joint[0:6] # left_leg
            obs[0, 45:51] = dq_joint[6:12] # right_leg

            # last_action
            obs[0, 51] = action[12] # joint_waist_yaw
            obs[0, 52:56] = action[13:17] # left_arm
            obs[0, 56:60] = action[17:21] # right_arm
            obs[0, 60:66] = action[0:6] # left_leg
            obs[0, 66:72] = action[6:12] # right_leg

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            # hist_obs.append(obs)
            # hist_obs.popleft()

            if init_state:
                import pdb
                # pdb.set_trace()
                policy_input[0, 0:cfg.env.num_single_obs] = obs[0, :cfg.env.num_single_obs]
                policy_input[0, cfg.env.num_single_obs:cfg.env.num_single_obs*2] = obs[0, :cfg.env.num_single_obs]
                policy_input[0, cfg.env.num_single_obs*2:cfg.env.num_single_obs*3] = obs[0, :cfg.env.num_single_obs]
                policy_input[0, cfg.env.num_single_obs*3:cfg.env.num_single_obs*4] = obs[0, :cfg.env.num_single_obs]
                policy_input[0, cfg.env.num_single_obs*4:cfg.env.num_single_obs*5] = obs[0, :cfg.env.num_single_obs]
                policy_input[0, cfg.env.num_single_obs*5:cfg.env.num_single_obs*6] = obs[0, :cfg.env.num_single_obs]
                init_state = False
            # for i in range(cfg.env.frame_stack):
                # policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            policy_input_tensor = torch.tensor(policy_input[0, cfg.env.num_single_obs:], dtype=torch.float32)

            obs_tensor = torch.tensor(obs[0], dtype=torch.float32)
 
            policy_input[0] = torch.cat((policy_input_tensor, obs_tensor), dim=0)

            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, cfg.normalization.clip_actions_min, cfg.normalization.clip_actions_max)
            print("action: ", action)

            action_mujoco = np.zeros_like(action)
            action_mujoco[0:6] = action[9:15]
            action_mujoco[6:12] = action[15:21]
            action_mujoco[12] = action[0]
            action_mujoco[13:17] = action[1:5]
            action_mujoco[17:21] = action[5:9]

            target_q = action_mujoco * cfg.control.action_scale# +  default_joint_angles

        tau = pd_control(target_q, q_joint, cfg.RobotConfig.kps,
                    target_dq, dq_joint, cfg.RobotConfig.kds)  # Calc torques
        tau = np.clip(tau, -cfg.RobotConfig.tau_limit, cfg.RobotConfig.tau_limit)
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1
    viewer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--robot_id', type=str, help='Path to the model to load.',default='gr1t2_simple')
    parser.add_argument('--load_model', type=str, required=True, help='Run to load from.',default="../policy/policy_3000.pt")
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
            decimation = 10
    
    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())

