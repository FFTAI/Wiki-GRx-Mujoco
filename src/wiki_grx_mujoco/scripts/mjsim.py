import os
import numpy
import torch
import argparse
import tqdm
import mujoco
import mujoco.viewer

from wiki_grx_mujoco.config.n1_config import N1Config

# log colors
WHITE = "\033[97m"
GRAY = "\033[90m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"


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
    model = mujoco.MjModel.from_xml_path(robot_cfg.sim.model_path)
    model.opt.timestep = robot_cfg.sim.dt
    data = mujoco.MjData(model)

    # Setup the viewer
    viewer = mujoco.viewer.launch_passive(model=model, data=data)

    # Prepare buffers
    target_q = numpy.zeros(robot_cfg.env.num_dofs, dtype=numpy.double)
    actions = numpy.zeros(robot_cfg.env.num_actions, dtype=numpy.double)

    decimation_count = 0
    gvec_tensor = torch.tensor([[0, 0, -1]], dtype=torch.float32)

    # Update the initial position
    mujoco.mj_step(model, data)

    # Calculate the total number of simulation steps
    total_steps = int(robot_cfg.sim.sim_duration / robot_cfg.sim.dt)

    # Print DOF sequence
    mj_joint_names = [model.jnt(i).name for i in range(model.njnt)]
    mj_joint_types = [model.jnt(i).type for i in range(model.njnt)]

    print(f"{WHITE}##################################################{RESET}")
    print(f"{WHITE}MUJOCO Joint Sequence and Names: {RESET}")
    for i in range(len(mj_joint_names)):
        print(f"{WHITE}  Joint {i}: {mj_joint_names[i]} {mj_joint_types[i]} {RESET}")
    print(f"{WHITE}##################################################{RESET}")

    q_base_names = mj_joint_names[0:1]
    q_dof_names = mj_joint_names[1:]

    q_obs_dof_names = [None] * robot_cfg.observation.num_dofs

    for i in range(robot_cfg.observation.num_dofs):
        index = robot_cfg.observation.index_dofs[i]

        q_obs_dof_names[i] = q_dof_names[index]

    print(f"{GREEN}##################################################{RESET}")
    print(f"{GREEN}MUJOCO q_obs_dof_names: {RESET}")
    for i in range(len(q_obs_dof_names)):
        print(f"{GREEN}  Joint {i}: {q_obs_dof_names[i]} {RESET}")
    print(f"{GREEN}##################################################{RESET}")

    q_dof_default = numpy.zeros(robot_cfg.env.num_dofs, dtype=numpy.double)
    q_obs_dof_default = numpy.zeros(robot_cfg.observation.num_dofs, dtype=numpy.double)

    for name, value in robot_cfg.init_state.default_joint_angles.items():
        if name in q_dof_names:
            index = q_dof_names.index(name)
            q_dof_default[index] = value
        else:
            pass

        if name in q_obs_dof_names:
            index = q_obs_dof_names.index(name)
            q_obs_dof_default[index] = value
        else:
            pass

    print(f"{YELLOW}##################################################{RESET}")
    print(f"{YELLOW}MUJOCO q_dof_default: {RESET}")
    for i in range(len(q_dof_default)):
        print(f"{YELLOW}  Joint {i}: {q_dof_names[i]} {q_dof_default[i]} {RESET}")
    print(f"{YELLOW}##################################################{RESET}")

    print(f"{YELLOW}##################################################{RESET}")
    print(f"{YELLOW}MUJOCO q_obs_dof_default: {RESET}")
    for i in range(len(q_obs_dof_default)):
        print(f"{YELLOW}  Joint {i}: {q_obs_dof_names[i]} {q_obs_dof_default[i]} {RESET}")
    print(f"{YELLOW}##################################################{RESET}")

    action_dof_names = [None] * robot_cfg.observation.num_actions

    for i in range(robot_cfg.observation.num_actions):
        index = robot_cfg.observation.index_actions[i]

        action_dof_names[i] = q_dof_names[index]

    action_scales = numpy.ones(robot_cfg.observation.num_actions, dtype=numpy.double)

    for i in range(robot_cfg.observation.num_actions):
        index = robot_cfg.observation.index_actions[i]

        q_name = q_dof_names[index]

        for j in range(len(robot_cfg.control.action_names)):
            action_name = robot_cfg.control.action_names[j]

            if action_name in q_name:
                action_scales[i] = robot_cfg.control.action_scale[action_name]
                break

    print(f"{BLUE}##################################################{RESET}")
    print(f"{BLUE}MUJOCO action info: {RESET}")
    for i in range(robot_cfg.observation.num_actions):
        print(f"{BLUE}  Action {i}: {action_dof_names[i]} {action_scales[i]} {RESET}")
    print(f"{BLUE}##################################################{RESET}")

    # Input
    policy_input = numpy.zeros([1, robot_cfg.env.num_stack_obs], dtype=numpy.float32)

    # ----------------------------------------------------------------------------------------------------

    # Iterate through each simulation step
    for step in tqdm.tqdm(range(total_steps), desc="Simulating..."):

        # Retrieve observation data
        q, dq, quat, omega = get_obs(data)

        q_base = q[0: 7]  # base_pos, base_quat (wxyz)
        dq_base = dq[0: 6]  # base_vel

        q_dof = q[7:]  # dof_pos
        dq_dof = dq[6:]  # dof_vel

        q_obs_dof = numpy.zeros(robot_cfg.observation.num_dofs, dtype=numpy.double)
        dq_obs_dof = numpy.zeros(robot_cfg.observation.num_dofs, dtype=numpy.double)

        for i in range(robot_cfg.observation.num_dofs):
            index = robot_cfg.observation.index_dofs[i]

            q_obs_dof[i] = q_dof[index]
            dq_obs_dof[i] = dq_dof[index]

        # RL policy
        if decimation_count % robot_cfg.sim.decimation == 0:
            obs = numpy.zeros([1, robot_cfg.env.num_obs], dtype=numpy.float32)

            # quat: mujoco wxyz -> pytorch xyzw
            quat_tensor = torch.from_numpy(numpy.array([quat])).float()
            quat_tensor = quat_tensor[:, [1, 2, 3, 0]]
            quat_proj = quat_rotate_inverse(quat_tensor, gvec_tensor)

            # omega: roll, pitch, yaw
            omega_tensor = torch.from_numpy(numpy.array([omega])).float()
            omega_proj = quat_rotate_inverse(quat_tensor, omega_tensor)
            # omega_proj = omega_tensor

            # q_obs_dof_offset
            q_obs_dof_offset = (q_obs_dof - q_obs_dof_default)

            # dq_obs_dofs
            dq_obs_dof = dq_obs_dof

            # actions
            actions = actions

            # obs
            obs[0, 0: 0 + 1] = robot_cfg.command.lin_vel_x
            obs[0, 1: 1 + 1] = robot_cfg.command.lin_vel_y
            obs[0, 2: 2 + 1] = robot_cfg.command.ang_vel_yaw
            obs[0, 3: 3 + 3] = omega_proj
            obs[0, 6: 6 + 3] = quat_proj
            obs[0, 9 + 0 * robot_cfg.observation.num_dofs: 9 + 1 * robot_cfg.observation.num_dofs] = q_obs_dof_offset * robot_cfg.normalization.obs_scales.dof_pos
            obs[0, 9 + 1 * robot_cfg.observation.num_dofs: 9 + 2 * robot_cfg.observation.num_dofs] = dq_obs_dof * robot_cfg.normalization.obs_scales.dof_vel
            obs[0, 9 + 2 * robot_cfg.observation.num_dofs: 9 + 2 * robot_cfg.observation.num_dofs + 1 * robot_cfg.observation.num_actions] = actions

            obs = numpy.clip(obs,
                             -robot_cfg.normalization.clip_observations,
                             +robot_cfg.normalization.clip_observations)

            # input
            policy_input[0, 0: robot_cfg.env.num_obs * (robot_cfg.env.num_stack - 1)] = policy_input[0, robot_cfg.env.num_obs:]
            policy_input[0, robot_cfg.env.num_obs * (robot_cfg.env.num_stack - 1):] = obs[0, :]

            # output
            output = policy.forward(torch.tensor(policy_input))

            # actions
            actions[:] = output[0].detach().numpy()
            actions = numpy.clip(actions,
                                 robot_cfg.normalization.clip_actions_min,
                                 robot_cfg.normalization.clip_actions_max)

            # actions scaled
            action_scaled = actions * action_scales

            # actions expanded
            actions_expanded = numpy.zeros(robot_cfg.env.num_dofs, dtype=numpy.float32)

            for i in range(robot_cfg.observation.num_actions):
                index = robot_cfg.observation.index_actions[i]

                actions_expanded[index] = action_scaled[i]

            # target DOFs
            target_q = actions_expanded + q_dof_default

            # Clear the decimation count
            decimation_count = 0

        # PD control
        tau = (target_q - q_dof) * robot_cfg.robot.kps + (0 - dq_dof) * robot_cfg.robot.kds
        tau = numpy.clip(tau,
                         -robot_cfg.robot.tau_limit,
                         +robot_cfg.robot.tau_limit)

        # Apply the control signal (torque control)
        data.ctrl = tau

        # Step the simulation
        mujoco.mj_step(model, data)
        viewer.sync()

        # Update the decimation count
        decimation_count += 1

    # ----------------------------------------------------------------------------------------------------

    # Close the viewer
    viewer.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--robot", type=str, required=True, help="Path to the model to load.")

    args = parser.parse_args()
    if args.robot == "N1":
        RobotConfig = N1Config
    else:
        raise ValueError(f"Unknown robot: {args.robot}")

    # Set up the robot configuration
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "robots",
        args.robot,
        "mjcf",
        "scene.xml"
    )

    print(
        "model_path = ", model_path
    )

    robot_cfg = RobotConfig()
    robot_cfg.sim.model_path = model_path

    # Load the policy
    policy_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "robots",
        args.robot,
        "policy",
        robot_cfg.policy.path
    )

    print(
        "policy_path = ", policy_path
    )

    if os.path.exists(policy_path):
        pass
    else:
        raise FileNotFoundError(f"File {policy_path} not found")

    policy = torch.jit.load(policy_path)

    # Run the Mujoco simulation
    run_mujoco(robot_cfg=robot_cfg, policy=policy)


if __name__ == "__main__":
    main()
