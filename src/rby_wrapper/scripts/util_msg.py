from rby_wrapper.msg import RobotState as RobotStateMsg
from builtin_interfaces.msg import Time
import rby1_sdk as rby
import numpy as np

def create_robot_state_msg(robot_state: rby.RobotState_A, model: rby.Model_A, gripper_state):
    """
    Create a RobotState message from robot state data and model indices.
    
    Args:
        robot_state: The robot state object containing position, velocity, current, torque, and target values
        model: The model object containing indices for different robot parts
        
    Returns:
        RobotStateMsg: A populated RobotState message
    """
    state_msg = RobotStateMsg()
    
    # Set timestamp
    state_msg.timestamp = Time()
    
    # Left arm state
    state_msg.left_arm_qpos = robot_state.position[model.left_arm_idx].tolist()
    state_msg.left_arm_vel = robot_state.velocity[model.left_arm_idx].tolist()
    state_msg.left_arm_current = robot_state.current[model.left_arm_idx].tolist()
    state_msg.left_arm_torque = robot_state.torque[model.left_arm_idx].tolist()
    state_msg.left_arm_target_qpos = robot_state.target_position[model.left_arm_idx].tolist()
    state_msg.left_arm_target_qvel = robot_state.target_velocity[model.left_arm_idx].tolist()
    
    # Right arm state
    state_msg.right_arm_qpos = robot_state.position[model.right_arm_idx].tolist()
    state_msg.right_arm_vel = robot_state.velocity[model.right_arm_idx].tolist()
    state_msg.right_arm_current = robot_state.current[model.right_arm_idx].tolist()
    state_msg.right_arm_torque = robot_state.torque[model.right_arm_idx].tolist()
    state_msg.right_arm_target_qpos = robot_state.target_position[model.right_arm_idx].tolist()
    state_msg.right_arm_target_qvel = robot_state.target_velocity[model.right_arm_idx].tolist()
    
    # Torso state
    state_msg.torso_qpos = robot_state.position[model.torso_idx].tolist()
    state_msg.torso_vel = robot_state.velocity[model.torso_idx].tolist()
    state_msg.torso_current = robot_state.current[model.torso_idx].tolist()
    state_msg.torso_torque = robot_state.torque[model.torso_idx].tolist()
    state_msg.torso_target_qpos = robot_state.target_position[model.torso_idx].tolist()
    state_msg.torso_target_qvel = robot_state.target_velocity[model.torso_idx].tolist()
    
    # Mobility (base) state
    state_msg.mobility_qpos = robot_state.position[model.mobility_idx].tolist()
    state_msg.mobility_vel = robot_state.velocity[model.mobility_idx].tolist()
    state_msg.mobility_current = robot_state.current[model.mobility_idx].tolist()
    state_msg.mobility_torque = robot_state.torque[model.mobility_idx].tolist()
    state_msg.mobility_target_qpos = robot_state.target_position[model.mobility_idx].tolist()
    state_msg.mobility_target_qvel = robot_state.target_velocity[model.mobility_idx].tolist()
    
    # Gripper state
    state_msg.left_gripper_qpos = gripper_state["left"]["qpos"]
    state_msg.left_gripper_target_qpos = gripper_state["left"]["target_qpos"]
    state_msg.left_gripper_min_qpos = gripper_state["left"]["min_qpos"]
    state_msg.left_gripper_max_qpos = gripper_state["left"]["max_qpos"]
    state_msg.left_gripper_qpos_normalized = normalize_position(state_msg.left_gripper_qpos, state_msg.left_gripper_min_qpos, state_msg.left_gripper_max_qpos)
    state_msg.left_gripper_target_qpos_normalized = normalize_position(state_msg.left_gripper_target_qpos, state_msg.left_gripper_min_qpos, state_msg.left_gripper_max_qpos)
    
    state_msg.right_gripper_qpos = gripper_state["right"]["qpos"]
    state_msg.right_gripper_target_qpos = gripper_state["right"]["target_qpos"]
    state_msg.right_gripper_min_qpos = gripper_state["right"]["min_qpos"]
    state_msg.right_gripper_max_qpos = gripper_state["right"]["max_qpos"]
    state_msg.right_gripper_qpos_normalized = normalize_position(state_msg.right_gripper_qpos, state_msg.right_gripper_min_qpos, state_msg.right_gripper_max_qpos)
    state_msg.right_gripper_target_qpos_normalized = normalize_position(state_msg.right_gripper_target_qpos, state_msg.right_gripper_min_qpos, state_msg.right_gripper_max_qpos)
    
    return state_msg

def normalize_position(position, min_position, max_position):
    """
    Normalize position values to range [0, 1] using min and max bounds.
    
    Args:
        position: The position value(s) to normalize (single value or list/array)
        min_position: The minimum position value(s) (single value or list/array)
        max_position: The maximum position value(s) (single value or list/array)
        
    Returns:
        Normalized position value(s) in range [0, 1]
    """
    
    # Check if inputs are scalar values
    is_scalar = np.isscalar(position) or (isinstance(position, (list, tuple)) and len(position) == 1)
    scalar_min = np.isscalar(min_position) or (isinstance(min_position, (list, tuple)) and len(min_position) == 1)
    scalar_max = np.isscalar(max_position) or (isinstance(max_position, (list, tuple)) and len(max_position) == 1)
    
    # Convert to numpy arrays for vectorized operations
    position_array = np.array(position)
    min_position_array = np.array(min_position)
    max_position_array = np.array(max_position)
    
    # Handle division by zero by checking for equal min and max
    range_values = max_position_array - min_position_array
    # Where range is zero, set it to 1 to avoid division by zero
    range_values = np.where(range_values == 0, 1.0, range_values)
    
    normalized = (position_array - min_position_array) / range_values
    
    # Return scalar if input was scalar
    if is_scalar and scalar_min and scalar_max:
        return float(normalized)
    return normalized

def denormalize_position(normalized_position, min_position, max_position):
    """
    Convert normalized position values [0, 1] back to original range.
    
    Args:
        normalized_position: The normalized position value(s) in range [0, 1] (single value or list/array)
        min_position: The minimum position value(s) in the original range (single value or list/array)
        max_position: The maximum position value(s) in the original range (single value or list/array)
        
    Returns:
        Position value(s) in the original range
    """
    # Check if inputs are scalar values
    is_scalar = np.isscalar(normalized_position) or (isinstance(normalized_position, (list, tuple)) and len(normalized_position) == 1)
    scalar_min = np.isscalar(min_position) or (isinstance(min_position, (list, tuple)) and len(min_position) == 1)
    scalar_max = np.isscalar(max_position) or (isinstance(max_position, (list, tuple)) and len(max_position) == 1)
    
    # Convert to numpy arrays
    norm_array = np.array(normalized_position)
    min_array = np.array(min_position)
    max_array = np.array(max_position)
    
    denormalized = norm_array * (max_array - min_array) + min_array
    
    # Return scalar if input was scalar
    if is_scalar and scalar_min and scalar_max:
        return float(denormalized)
    return denormalized