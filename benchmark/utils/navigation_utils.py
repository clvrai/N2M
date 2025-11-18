"""Navigation utilities for benchmark.

Uses Mujoco low-level API for teleportation (no PID control).
Migrated from A_ref/N2M-sim/robomimic/robomimic/utils/train_utils.py
"""

import numpy as np
from benchmark.utils.transform_utils import qpos_command_wrapper, obs_to_SE2


def teleport_robot_to_pose(env, target_pose: np.ndarray):
    """Teleport robot to target pose using Mujoco low-level API.
    
    NO PID control - directly set qpos and step simulator.
    
    Args:
        env: RoboCasa environment
        target_pose: SE2 pose [x, y, theta]
    """
    # Get unwrapped environment
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env.env
    robot = unwrapped_env.robots[0]
    
    # Set base joint positions (wrap command for mujoco x-axis flip)
    unwrapped_env.sim.data.qpos[robot._ref_base_joint_pos_indexes] = \
        qpos_command_wrapper(target_pose)
    
    # Forward simulate to update state
    unwrapped_env.sim.forward()
    
    # Step environment with zero action to update observations
    zero_action = np.zeros(env.action_dim)
    env.step(zero_action)


def get_current_robot_pose(env) -> np.ndarray:
    """Get current robot base SE2 pose.
    
    Args:
        env: RoboCasa environment
        
    Returns:
        pose: SE2 pose [x, y, theta]
    """
    obs = env.get_observation()
    return obs_to_SE2(obs)


def move_robot_away(env, far_pose: np.ndarray = np.array([0.0, -50.0, 0.0])):
    """Move robot to a far-away position (useful for data collection).
    
    Args:
        env: RoboCasa environment
        far_pose: Far away SE2 pose (default [0.0, -50.0, 0.0])
    """
    teleport_robot_to_pose(env, far_pose)
