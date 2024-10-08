import numpy as np
import mujoco
import mujoco_viewer
from tqdm import tqdm
from run.robots.robot_config.GR1_mj_config import GR1T1LowerLimbCfg
from run.robots.robot_config.GR2_mj_config import GR1T2LowerLimbCfg
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
    
    # Listener setup
    def on_press(key):
        nonlocal policy
        try:
            if key.char == '.':
                policy = torch.jit.load('../policy/stand_model_jit.pt')
            elif key.char == '/':
                policy = torch.jit.load('../policy/walk_model_jit.pt')
            elif key.char == '0':
                return False  # Stop the listener
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        q, dq, quat, omega = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        joint_names = ['l_hip_roll',
                       'l_hip_yaw',
                       'l_hip_pitch',
                       'l_knee_pitch',
                       'l_ankle_pitch',
                       'r_hip_roll',
                       'r_hip_yaw',
                       'r_hip_pitch',
                       'r_knee_pitch',
                       'r_ankle_pitch']
        
        default_joint_angles = np.array([cfg.init_state.default_joint_angles[name] for name in joint_names])
        default_joint_angles = default_joint_angles[-cfg.env.num_actions:]

        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            quat_tensor = torch.tensor([quat], dtype=torch.float32)
            quat_tensor = quat_tensor[:, [1, 2, 3, 0]]
            omega_tensor = torch.tensor([omega], dtype=torch.float32)
            quat_proj = quat_rotate_inverse(quat_tensor, gvec_tensor)
            omega_proj = quat_rotate_inverse(quat_tensor, omega_tensor)

            obs[0, 0:3] = omega_proj
            obs[0, 3:6] = quat_proj
            obs[0, 6] = cmd.vx
            obs[0, 7] = cmd.vy
            obs[0, 8] = cmd.dyaw
            obs[0, 9:19] = (q - default_joint_angles) * cfg.normalization.obs_scales.dof_pos
            obs[0, 19:29] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 29:39] = action

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            policy_input[0, :cfg.env.num_single_obs] = obs[0, :cfg.env.num_single_obs]

            action[:] = policy.forward(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, cfg.normalization.clip_actions_min, cfg.normalization.clip_actions_max)

            target_q = (action + default_joint_angles) * cfg.control.action_scale
            
        tau = (target_q - q) * cfg.RobotConfig.kps - dq * cfg.RobotConfig.kds
        tau = np.clip(tau, -cfg.RobotConfig.tau_limit, cfg.RobotConfig.tau_limit)
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1
    viewer.close()
    listener.stop()  # Ensure the listener is stopped after the simulation ends

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

    class Sim2simCfg(RobotConfig):
        class sim_config:
            mujoco_model_path = f'../robots/{args.robot_id}/scene.xml'
            sim_duration = 70.0
            dt = 0.001
            decimation = 20
    
    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
