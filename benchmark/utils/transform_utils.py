"""Coordinate transformation utilities.

Migrated from A_ref/N2M-sim/robomimic/robomimic/utils/navi_utils.py
and misc_util.py
"""

import numpy as np
from scipy.spatial.transform import Rotation


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def obs_to_SE2(observation: dict, algorithm_name: str = None) -> np.ndarray:
    """Convert observation dictionary to SE2 pose [x, y, theta].
    
    Args:
        observation: Observation dict containing robot0_base_pos and robot0_base_quat
        algorithm_name: Algorithm name (e.g., 'act'), affects how observation is indexed
        
    Returns:
        se2: np.ndarray of shape (3,) representing [x, y, theta]
    """
    pos = observation['robot0_base_pos']
    quat = observation['robot0_base_quat']
    
    # Handle different observation formats
    if isinstance(pos, np.ndarray) and pos.ndim > 1:
        pos = pos[-1]  # Take last timestep if batched
    if isinstance(quat, np.ndarray) and quat.ndim > 1:
        quat = quat[-1]  # Take last timestep if batched
    
    # Ensure they are arrays
    pos = np.atleast_1d(pos)
    quat = np.atleast_1d(quat)
    
    # Convert quaternion to yaw
    yaw = np.arctan2(
        2 * (quat[3] * quat[2] + quat[0] * quat[1]), 
        1 - 2 * (quat[1]**2 + quat[2]**2)
    )
    se2 = np.array([pos[0], pos[1], yaw])
    return se2


def obs_to_SE3(observation: dict) -> np.ndarray:
    """Convert observation dictionary to SE3 pose as 4x4 transformation matrix.
    
    Args:
        observation: Observation dict containing robot0_base_pos and robot0_base_quat
        
    Returns:
        T: np.ndarray of shape (4, 4) representing SE3 transformation matrix
    """
    pos = observation['robot0_base_pos'][-1]  # [x, y, z]
    quat = observation['robot0_base_quat'][-1]  # [x, y, z, w]
    
    # Convert quaternion to rotation matrix
    x, y, z, w = quat
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    
    return T


def qpos_command_wrapper(command: np.ndarray) -> np.ndarray:
    """Wrap qpos command for Mujoco (flip x axis).
    
    Since mujoco x axis is flipped, we need to flip the x axis of the command.
    
    Args:
        command: SE2 command [x, y, theta]
        
    Returns:
        new_command: Wrapped command with flipped x axis
    """
    new_command = np.zeros_like(command)
    new_command[0] = -command[0]
    new_command[1] = command[1]
    new_command[2] = command[2]
    return new_command


def convert_extrinsic_to_pos_and_quat(extrinsic: np.ndarray) -> tuple:
    """Convert 4x4 extrinsic matrix to position and quaternion.
    
    Args:
        extrinsic: 4x4 extrinsic transformation matrix
        
    Returns:
        pos: Position [x, y, z]
        quat_wxyz: Quaternion in [w, x, y, z] format
    """
    pos = extrinsic[:3, 3]
    rot = Rotation.from_matrix(extrinsic[:3, :3])
    quat_xyzw = rot.as_quat()  # scipy returns [x,y,z,w]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos, quat_wxyz


def se2_to_se3(se2: np.ndarray, z: float = 0.0) -> np.ndarray:
    """Convert SE2 pose to SE3 transformation matrix.
    
    Args:
        se2: SE2 pose [x, y, theta]
        z: Height (default 0.0)
        
    Returns:
        T: 4x4 SE3 transformation matrix
    """
    x, y, theta = se2
    
    # Rotation around z-axis
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # Build SE3 matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T
