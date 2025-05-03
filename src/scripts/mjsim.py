import os
import numpy
import torch
import argparse
import tqdm
import mujoco
import mujoco.viewer

from src.robots.robot_config.GR1T1_mj_config import GR1T1LowerLimbCfg
from src.robots.robot_config.GR1T2_mj_config import GR1T2LowerLimbCfg
from src.robots.N1.config.mj_config import N1Config


# Define the Command class
class Command:
    lin_vel_x = 0.25
    lin_vel_y = 0
    ang_vel_yaw = 0


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
    q = data.qpos.astype(numpy.double)
    dq = data.qvel.astype(numpy.double)
    quat = data.sensor("orientation").data.astype(numpy.double)
    omega = data.sensor("angular-velocity").data.astype(numpy.double)
    return q, dq, quat, omega


# Function to src the Mujoco simulation
def run_mujoco(
        robot_cfg,
        policy,
) -> None:
    """
    Run Mujoco simulation with the given robot configuration and policy.

    :param robot_cfg:
    :param policy:
    :return None:
    """

    # Load the Mujoco model
    model = mujoco.MjModel.from_xml_path(robot_cfg.sim_config.mujoco_model_path)
    model.opt.timestep = robot_cfg.sim_config.dt
    data = mujoco.MjData(model)

    # Setup the viewer
    viewer = mujoco.viewer.launch_passive(model=model, data=data)

    # Prepare buffers
    target_q = numpy.zeros(robot_cfg.env.num_actions, dtype=numpy.double)
    action = numpy.zeros(robot_cfg.env.num_actions, dtype=numpy.double)

    decimation_count = 0
    gvec_tensor = torch.tensor([[0, 0, -1]], dtype=torch.float32)

    # Update the initial position
    mujoco.mj_step(model, data)

    # Calculate the total number of simulation steps
    total_steps = int(robot_cfg.sim_config.sim_duration / robot_cfg.sim_config.dt)

    # Iterate through each simulation step
    for step in tqdm.tqdm(range(total_steps), desc="Simulating..."):

        # Retrieve observation data
        q, dq, quat, omega = get_obs(data)
        q = q[-robot_cfg.env.num_actions:]
        dq = dq[-robot_cfg.env.num_actions:]

        joint_names = [
            "l_hip_roll",
            "l_hip_yaw",
            "l_hip_pitch",
            "l_knee_pitch",
            "l_ankle_pitch",
            "r_hip_roll",
            "r_hip_yaw",
            "r_hip_pitch",
            "r_knee_pitch",
            "r_ankle_pitch",
        ]

        default_joint_angles = numpy.array([robot_cfg.init_state.default_joint_angles[name] for name in joint_names])
        default_joint_angles = default_joint_angles[-robot_cfg.env.num_actions:]

        # RL policy
        if decimation_count % robot_cfg.sim_config.decimation == 0:
            obs = numpy.zeros([1, robot_cfg.env.num_single_obs], dtype=numpy.float32)

            # quat: mujoco wxyz -> pytorch xyzw
            quat_tensor = torch.from_numpy(numpy.array([quat])).float()
            quat_tensor = quat_tensor[:, [1, 2, 3, 0]]
            quat_proj = quat_rotate_inverse(quat_tensor, gvec_tensor)

            # omega: roll, pitch, yaw
            omega_tensor = torch.from_numpy(numpy.array([omega])).float()
            omega_proj = quat_rotate_inverse(quat_tensor, omega_tensor)

            # q_offset
            q_offset = (q - default_joint_angles)

            # obs
            obs[0, 0:3] = omega_proj
            obs[0, 3:6] = quat_proj
            obs[0, 6] = Command.lin_vel_x
            obs[0, 7] = Command.lin_vel_y
            obs[0, 8] = Command.ang_vel_yaw
            obs[0, 9:19] = q_offset * robot_cfg.normalization.obs_scales.dof_pos
            obs[0, 19:29] = dq * robot_cfg.normalization.obs_scales.dof_vel
            obs[0, 29:39] = action

            obs = numpy.clip(obs,
                             -robot_cfg.normalization.clip_observations,
                             +robot_cfg.normalization.clip_observations)

            # input
            policy_input = numpy.zeros([1, robot_cfg.env.num_observations], dtype=numpy.float32)
            policy_input[0, :robot_cfg.env.num_single_obs] = obs[0, :robot_cfg.env.num_single_obs]

            # output
            output = policy.forward(torch.tensor(policy_input))

            # action
            action[:] = output[0].detach().numpy()
            action = numpy.clip(action,
                                robot_cfg.normalization.clip_actions_min,
                                robot_cfg.normalization.clip_actions_max)
            action_scaled = action * robot_cfg.control.action_scale

            target_q = action_scaled + default_joint_angles

            # Clear the decimation count
            decimation_count = 0

        # PD control
        tau = (target_q - q) * robot_cfg.RobotConfig.kps + (0 - dq) * robot_cfg.RobotConfig.kds
        tau = numpy.clip(tau, -robot_cfg.RobotConfig.tau_limit, robot_cfg.RobotConfig.tau_limit)

        # Apply the control signal (torque control)
        data.ctrl = tau

        # Step the simulation
        mujoco.mj_step(model, data)
        viewer.sync()

        # Update the decimation count
        decimation_count += 1

    viewer.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--robot", type=str, help="Path to the model to load.")
    parser.add_argument("--policy", type=str, required=True, help="Run to load from.")
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")

    args = parser.parse_args()
    if args.robot == "N1":
        RobotConfig = N1Config
    else:
        raise ValueError(f"Unknown robot: {args.robot}")

    # Set up the robot configuration
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "robots",
        args.robot,
        "scene.xml"
    )


    class Sim2SimCfg(RobotConfig):
        class sim_config:
            mujoco_model_path = model_path
            sim_duration = 60.0  # seconds
            dt = 0.001  # seconds
            decimation = 20  # decimation factor


    robot_cfg = Sim2SimCfg()

    # Load the policy
    policy_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "policy",
        args.policy
    )

    if os.path.exists(policy_path):
        print(f"Loading policy from {policy_path}")
    else:
        raise FileNotFoundError(f"File {policy_path} not found")

    policy = torch.jit.load(policy_path)

    # Run the Mujoco simulation
    run_mujoco(robot_cfg=robot_cfg, policy=policy)
