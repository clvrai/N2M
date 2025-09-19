import rby1_sdk as rby
from typing import Union
from util_pose import Pose
from util_settings import Settings
import numpy as np

class BaseCmdData:
    def __init__(self, vel=0.0, omega=0.0):
        self.vel = vel     # linear velocity
        self.omega = omega # angular velocity

class WBCCommand:
    def __init__(self, pose: Pose):
        self.left_arm_qpos = pose.left_arm
        self.right_arm_qpos = pose.right_arm
        self.torso_qpos = pose.torso
        self.base_cmd = BaseCmdData(vel=0.0, omega=0.0)
        self.left_gripper_qpos_normalized = 0.0
        self.right_gripper_qpos_normalized = 0.0

# Safety function to limit joint movement
def limit_joint_movement(current_pos, target_pos, max_delta_rad=0.1):
    """Limit joint movement to prevent damage to servos"""
    current_pos = np.array(current_pos)
    target_pos = np.array(target_pos)
    delta = target_pos - current_pos
    # Calculate magnitude of each joint movement
    delta_mag = np.abs(delta)
    # Find joints that exceed the limit
    exceeded = delta_mag > max_delta_rad
    if np.any(exceeded):
        # Scale down movements that exceed the limit
        scale_factors = np.ones_like(delta)
        scale_factors[exceeded] = max_delta_rad / delta_mag[exceeded]
        # Apply scaling to limit movement
        limited_delta = delta * scale_factors
        # Return the limited target position
        return current_pos + limited_delta
    return target_pos
    
def joint_position_command_builder_safe(
    pose: Pose = None, safety_dict=None, base_cmd: BaseCmdData = None, minimum_time=1/30, model:rby.Model_A=None, control_hold_time=1e6,
):
    """
    Build a joint position command with safety constraints from the safety_dict.
    
    Args:
        pose: The target pose
        base_cmd: Optional base command with linear velocity and angular velocity (default: None)
        minimum_time: Minimum time to complete the movement
        safety_dict: Dictionary containing safety limits (robot_max_q, robot_min_q, robot_max_qdot, robot_max_qddot)
        model: Robot model containing joint indices
        control_hold_time: Time to hold the control (default: 1e6)
    """
    # Extract safety limits
    robot_max_q = safety_dict["robot_max_q"]
    robot_min_q = safety_dict["robot_min_q"]
    robot_max_qdot = safety_dict["robot_max_qdot"]
    robot_max_qddot = safety_dict["robot_max_qddot"]
    base_max_acc = safety_dict["base_max_acc"]
    base_max_vel = safety_dict["base_max_vel"]
    base_max_omega = safety_dict["base_max_omega"]
    
    wbc_command =  rby.ComponentBasedCommandBuilder()
    body_command_builder = rby.BodyComponentBasedCommandBuilder()
    # Right arm with safety constraints
    right_arm_builder = rby.JointPositionCommandBuilder()
    (
        right_arm_builder.set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
        )
        .set_position(
            np.clip(
                pose.right_arm,
                robot_min_q[model.right_arm_idx],
                robot_max_q[model.right_arm_idx],
            )
        )
        .set_velocity_limit(robot_max_qdot[model.right_arm_idx])
        .set_acceleration_limit(robot_max_qddot[model.right_arm_idx] * 30)
        .set_minimum_time(minimum_time)
    )
    body_command_builder.set_right_arm_command(right_arm_builder)
    
    # Left arm with safety constraints
    left_arm_builder = rby.JointPositionCommandBuilder()
    (
        left_arm_builder.set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
        )
        .set_position(
            np.clip(
                pose.left_arm,
                robot_min_q[model.left_arm_idx],
                robot_max_q[model.left_arm_idx],
            )
        )
        .set_velocity_limit(robot_max_qdot[model.left_arm_idx])
        .set_acceleration_limit(robot_max_qddot[model.left_arm_idx] * 30)
        .set_minimum_time(minimum_time)
    )
    body_command_builder.set_left_arm_command(left_arm_builder)
    
    # Torso with safety constraints
    torso_builder = rby.JointPositionCommandBuilder()
    (
        torso_builder.set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
        )
        .set_position(
            np.clip(
                pose.torso,
                robot_min_q[model.torso_idx],
                robot_max_q[model.torso_idx],
            )
        )
        .set_velocity_limit(robot_max_qdot[model.torso_idx])
        .set_acceleration_limit(robot_max_qddot[model.torso_idx] * 30)
        .set_minimum_time(minimum_time)
    )
    body_command_builder.set_torso_command(torso_builder)
    
    base_command_builder = None
    # Add mobility command if base_cmd is provided
    if base_cmd is not None:
        clip_vel = np.clip(base_cmd.vel, -base_max_vel, base_max_vel)
        clip_omega = np.clip(base_cmd.omega, -base_max_omega, base_max_omega)
        # print(f"clip_vel: {clip_vel}, clip_omega: {clip_omega}, acc: {base_max_acc}")
        base_command_builder = rby.SE2VelocityCommandBuilder()
        base_command_builder.set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)) 
        base_command_builder.set_minimum_time(minimum_time)
        base_command_builder.set_acceleration_limit(np.array([base_max_acc, base_max_acc]), base_max_acc)
        base_command_builder.set_velocity(np.array([clip_vel, 0]), clip_omega)  # linear velocity, angular velocity
        return wbc_command.set_body_command(body_command_builder).set_mobility_command(base_command_builder)
    else:
        return wbc_command.set_body_command(body_command_builder)

def joint_position_command_builder(
    pose: Pose, minimum_time, control_hold_time=0, position_mode=True,
):
    right_arm_builder = (
        rby.JointPositionCommandBuilder()
        if position_mode
        else rby.JointImpedanceControlCommandBuilder()
    )
    (
        right_arm_builder.set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
        )
        .set_position(pose.right_arm)
        .set_minimum_time(minimum_time)
    )
    if not position_mode:
        (
            right_arm_builder.set_stiffness(
                [Settings.impedance_stiffness] * len(pose.right_arm)
            )
            .set_damping_ratio(Settings.impedance_damping_ratio)
            .set_torque_limit([Settings.impedance_torque_limit] * len(pose.right_arm))
        )

    left_arm_builder = (
        rby.JointPositionCommandBuilder()
        if position_mode
        else rby.JointImpedanceControlCommandBuilder()
    )
    (
        left_arm_builder.set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
        )
        .set_position(pose.left_arm)
        .set_minimum_time(minimum_time)
    )
    if not position_mode:
        (
            left_arm_builder.set_stiffness(
                [Settings.impedance_stiffness] * len(pose.left_arm)
            )
            .set_damping_ratio(Settings.impedance_damping_ratio)
            .set_torque_limit([Settings.impedance_torque_limit] * len(pose.left_arm))
        )

    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_torso_command(
                rby.JointPositionCommandBuilder()
                .set_command_header(
                    rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
                )
                .set_position(pose.torso)
                .set_minimum_time(minimum_time)
            )
            .set_right_arm_command(right_arm_builder)
            .set_left_arm_command(left_arm_builder)
        )
    )

def move_j(
    robot: Union[rby.Robot_A, rby.Robot_T5, rby.Robot_M], pose: Pose, minimum_time=5.0
):
    handler = robot.send_command(joint_position_command_builder(pose, minimum_time))
    return handler.get() == rby.RobotCommandFeedback.FinishCode.Ok

def move_j_safe(
    robot: Union[rby.Robot_A, rby.Robot_T5, rby.Robot_M], 
    pose: Pose, 
    safety_dict,
    model,
    base_cmd = None,
    minimum_time=5.0
):
    wbc_command = joint_position_command_builder_safe(
        pose=pose, 
        safety_dict=safety_dict, 
        base_cmd=base_cmd,
        minimum_time=minimum_time, 
        model=model, 
        control_hold_time=0.0
    )
    handler = robot.send_command(wbc_command)
    return handler.get() == rby.RobotCommandFeedback.FinishCode.Ok