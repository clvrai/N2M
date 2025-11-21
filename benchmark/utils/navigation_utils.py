"""Navigation utilities for benchmark.

Uses Mujoco low-level API for teleportation (no PID control).
Migrated from A_ref/N2M-sim/robomimic/robomimic/utils/train_utils.py
"""

import numpy as np
from benchmark.utils.transform_utils import qpos_command_wrapper, obs_to_SE2


def teleport_robot_to_target(unwrapped_env, se2_target: np.ndarray, se2_initial: np.ndarray = np.array([0.0, 0.0, 0.0])):
    """Teleport robot to target pose using Mujoco low-level API.
        robocasa use se2_initial as their coordinate origin (position and direction)
        both se2_target and se2_initial are in the global coordinate system
        need first transform se2_target to the local coordinate system, then set the qpos
    """
    robot = unwrapped_env.robots[0]

    offset = 0.21
    se2_initial = se2_initial - offset * np.array([np.cos(se2_initial[2]), np.sin(se2_initial[2]), 0])
    se2_target = se2_target - offset * np.array([np.cos(se2_target[2]), np.sin(se2_target[2]), 0])

    # Transform se2_target from global frame to se2_initial's local frame
    # SE2 transformation: rotate translation by -theta_initial and subtract
    dx_global = se2_target[0] - se2_initial[0]
    dy_global = se2_target[1] - se2_initial[1]
    theta_initial = se2_initial[2]
    
    # Rotate global translation to local frame
    cos_theta = np.cos(theta_initial)
    sin_theta = np.sin(theta_initial)
    dx_local = dx_global * cos_theta + dy_global * sin_theta
    dy_local = -dx_global * sin_theta + dy_global * cos_theta
    dtheta_local = se2_target[2] - se2_initial[2]
    
    relative_se2 = np.array([dx_local, dy_local, dtheta_local])
    unwrapped_env.sim.data.qpos[robot._ref_base_joint_pos_indexes] = relative_se2
    
    # Forward simulate to update state
    unwrapped_env.sim.forward()


def get_current_robot_pose(env) -> np.ndarray:
    """Get current robot base SE2 pose.
    
    Args:
        env: RoboCasa environment
        
    Returns:
        pose: SE2 pose [x, y, theta]
    """
    obs = env.get_observation()
    return obs_to_SE2(obs)
