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
import time
from collections import deque
from scipy.spatial.transform import Rotation as R

# import IPython
# e = IPython.embed

# Define the cmd class
class cmd:
    vx = 0.4
    vy = 0.0
    dyaw = 0.0

# Function to rotate quaternion inversely
def quat_rotate_inverse(q, v):
    q_vec = q[:, :3]  # Vector part (x, y, z)
    q_w = q[:, -1]    # Scalar part (w)
    
    # Quaternion rotation formula (inverse rotation by q^-1):
    a = 2.0 * torch.sum(q_vec * v, dim=-1, keepdim=True) * q_vec  # Dot product part
    b = q_w.unsqueeze(-1) * torch.cross(q_vec, v, dim=-1) * 2.0   # Cross product part
    
    return v - a + b

# # Function to get observation data
# def get_obs(data):
#     q = data.qpos.astype(np.double)
#     dq = data.qvel.astype(np.double)
#     quat = data.sensor('orientation').data.astype(np.double)
#     omega = data.sensor('angular-velocity').data.astype(np.double)
#     return (q, dq, quat, omega)

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

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
    # init_state = True
    init_time = 0

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


    hist_obs = deque()
    for _ in range(6):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))


    
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        '''
        q: position observation
        dq: speed observation
        quat: quaternion -> orientation observation, from imu
        omega: angular velocity observation, from imu
        '''

        start_time = time.time()
        # q, dq, quat, omega = get_obs(data)
        q, dq, quat, v, omega, gvec = get_obs(data)
        # print(action)


        if count_lowlevel < 150:
            target_q = default_joint_angles
            # target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # 28,27, 4, 3

        # for free flyer: q:7, dq:6
        q_base = q[0:3]
        dq_base = dq[0:3]
        quat = q[3:7]
        omega = dq[3:6]
        
        # for rest joint: 21
        q_joint = q[7:]
        dq_joint = dq[6:]

        
        if count_lowlevel % cfg.sim_config.decimation == 0 and count_lowlevel >= 150:
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32) # 75
            # quat_tensor = torch.tensor([quat], dtype=torch.float32)
            # quat_tensor = quat_tensor[:, [1, 2, 3, 0]]
            # omega_tensor = torch.tensor([omega], dtype=torch.float32)
            gravity = model.opt.gravity
            # gravity_tensor = torch.tensor([gravity], dtype=torch.float32)
            # quat_proj = quat_rotate_inverse(quat_tensor, gvec_tensor)
            # omega_proj = quat_rotate_inverse(quat_tensor, omega_tensor)
            # gravity_proj = quat_rotate_inverse(quat_tensor, gravity_tensor)

            # print(omega_tensor)


            # import pdb 
            # pdb.set_trace()

            obs[0, 0] = cmd.vx
            obs[0, 1] = cmd.vy
            obs[0, 2] = cmd.dyaw

            obs[0, 3:6] = omega # (base angular velocity)
            # print(omega)

            

            # Gravity components, x, y, z
            obs[0, 6:9] = gravity 
            # obs[0, 7] = gravity_proj[1]
            # obs[0, 8] = gravity_proj[2]
            # print(gravity)
            # obs[0, 8] = -98

            # position
            obs[0, 9] = q_joint[12] # joint_waist_yaw
            obs[0, 10:14] = q_joint[13:17] # left_arm
            obs[0, 14:18] = q_joint[17:21] # right_arm
            # obs[0,15] = - obs[0,15]
            obs[0, 18:24] = q_joint[0:6] # left_leg
            obs[0, 24:30] = q_joint[6:12] # right_leg

            # velocity
            obs[0, 30] = dq_joint[12] # joint_waist_yaw
            obs[0, 31:35] = dq_joint[13:17] # left_arm
            obs[0, 35:39] = dq_joint[17:21] # right_arm
            # obs[0,36] = - obs[0,36]
            obs[0, 39:45] = dq_joint[0:6] # left_leg
            obs[0, 45:51] = dq_joint[6:12] # right_leg


            # last_action
            obs[0, 51] = action[12] # joint_waist_yaw
            obs[0, 52:56] = action[13:17] # left_arm
            obs[0, 56:60] = action[17:21] # right_arm
            obs[0, 60:66] = action[0:6] # left_leg
            obs[0, 66:72] = action[6:12] # right_leg
            # print(obs, '\n')



            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()


            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(6):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()

            # action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            # target_q = action * cfg.control.action_scale



            # if init_state:
            #     policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            #     policy_input[0, 0:cfg.env.num_single_obs] = obs[0, :cfg.env.num_single_obs]
            #     policy_input[0, cfg.env.num_single_obs:cfg.env.num_single_obs*2] = obs[0, :cfg.env.num_single_obs]
            #     policy_input[0, cfg.env.num_single_obs*2:cfg.env.num_single_obs*3] = obs[0, :cfg.env.num_single_obs]
            #     policy_input[0, cfg.env.num_single_obs*3:cfg.env.num_single_obs*4] = obs[0, :cfg.env.num_single_obs]
            #     policy_input[0, cfg.env.num_single_obs*4:cfg.env.num_single_obs*5] = obs[0, :cfg.env.num_single_obs]
            #     policy_input[0, cfg.env.num_single_obs*5:cfg.env.num_single_obs*6] = obs[0, :cfg.env.num_single_obs]
            #     init_state = False
            # if not init_state:
            #     policy_input[0, 0:cfg.env.num_single_obs*5] = policy_input[0, 72:] 
            #     policy_input[0, cfg.env.num_single_obs*5:cfg.env.num_single_obs*6] = obs[0, :cfg.env.num_single_obs]

            # action[:] = policy.forward(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, cfg.normalization.clip_actions_min, cfg.normalization.clip_actions_max)

            target_q = default_joint_angles + action * cfg.control.action_scale
            
            # target_q = action
            # print(action)
            # print(dq_joint)
            # print(target_q, '\n')

            
        # print(target_q)
        # print(q_joint)
        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

        # target_q = default_joint_angles
        tau = (target_q - q_joint) * cfg.RobotConfig.kps - dq_joint * cfg.RobotConfig.kds
        # tau = np.clip(tau, -cfg.RobotConfig.tau_limit, cfg.RobotConfig.tau_limit)
        # tau[0] = -5
        data.ctrl = tau
        
        # zero_output = np.zeros_like(tau)
        # zero_output[21] = 10
       
        # data.ctrl = zero_output
        
        print(tau, "tau\n")
        mujoco.mj_step(model, data)
        viewer.render()
        hz = 1/(time.time()-start_time)
        # print(hz, 'hz')
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
            gravity = [0,0,-9.81]
    
    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())

