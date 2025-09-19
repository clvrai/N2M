#!/usr/bin/env python3

"""
Teleoperation Example

Run this example on UPC to which the master arm and hands are connected
"""

import rby1_sdk as rby
import numpy as np
import os
import time
import logging
import argparse
import signal
import threading
from typing import *
from dataclasses import dataclass
from copy import deepcopy
from rby_wrapper.msg import RobotState as RobotStateMsg
from util_msg import create_robot_state_msg
from std_msgs.msg import Bool, Int32
from util_components import Gripper
from util_command import joint_position_command_builder, joint_position_command_builder_safe, move_j, move_j_safe, Settings, limit_joint_movement
from util_pose import INIT_POSE, MANIPULATION_POSE, MANIPULATION_POSE_DICT,Pose

from geometry_msgs.msg import Twist
from rby_wrapper.msg import RobotControl as RobotControlMsg
from util_command import WBCCommand, BaseCmdData

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def main(address, model, power, servo, control_mode):
    # ===== Choose position Index =====
    print("Choose position index:")
    for key, item in MANIPULATION_POSE_DICT.items():
        print(f"{key}: {item}")
    position_idx = int(input("Enter position index:"))
    
    # ===== SETUP ROBOT =====
    robot = rby.create_robot(address, model)
    if not robot.connect():
        logging.error(f"Failed to connect robot {address}")
        exit(1)
    supported_model = ["A", "T5", "M"]
    supported_control_mode = ["position", "impedance"]
    model = robot.model()
    dyn_model = robot.get_dynamics()
    dyn_state = dyn_model.make_state([], model.robot_joint_names)
    robot_q = None
    robot_state = None
    
    # ===== SETUP SAFETY DICT =====
    robot_max_q = dyn_model.get_limit_q_upper(dyn_state)
    robot_min_q = dyn_model.get_limit_q_lower(dyn_state)
    robot_max_qdot = dyn_model.get_limit_qdot_upper(dyn_state)
    robot_max_qddot = dyn_model.get_limit_qddot_upper(dyn_state)
    
    robot_max_qdot[model.right_arm_idx[-1]] *= 10  # wrist joint can be faster
    robot_max_qdot[model.left_arm_idx[-1]] *= 10  # wrist joint can be faster
    robot_max_qdot[model.left_arm_idx] *= 0.3  
    robot_max_qdot[model.right_arm_idx] *= 0.3
    robot_max_qdot[model.torso_idx] *= 0.08
    robot_max_qddot *= 0.3  # slow down all joints for safety
    
    safety_dict = {
        "robot_max_q": robot_max_q,
        "robot_min_q": robot_min_q,
        "robot_max_qdot": robot_max_qdot,
        "robot_max_qddot": robot_max_qddot,
        "base_max_acc": 0.3,
        "base_max_vel": 0.5,
        "base_max_omega":0.5
    }
    
    # ======== Global Vairables ==========
    # Create thread-safe variables for episode status
    episode_lock = threading.Lock()
    episode_done = False
    episode_start = False
    base_cmd = BaseCmdData(0.0, 0.0)
    last_base_time = None
    
    if not model.model_name in supported_model:
        logging.error(
            f"Model {model.model_name} not supported (Current supported model is {supported_model})"
        )
        exit(1)
    if not control_mode in supported_control_mode:
        logging.error(
            f"Control mode {control_mode} not supported (Current supported control mode is {supported_control_mode})"
        )
        exit(1)
    position_mode = control_mode == "position"
    if not robot.is_power_on(power):
        if not robot.power_on(power):
            logging.error(f"Failed to turn power ({power}) on")
            exit(1)
            
    # Make sure wheel servos are on for base movement
    if not robot.is_servo_on(".*(right_wheel|left_wheel).*"):
        if not robot.servo_on(".*(right_wheel|left_wheel).*"):
            logging.error("Failed to servo on wheels - Base movement will not work")
            logging.warning("Continuing without wheel control...")
    logging.info(f"Enabled wheel servos, be careful with base movement")
        
    if not robot.is_servo_on(servo):
        if not robot.servo_on(servo):
            logging.error(f"Failed to servo ({servo}) on")
            exit(1)
    robot.reset_fault_control_manager()
    if not robot.enable_control_manager():
        logging.error(f"Failed to enable control manager")
        exit(1)
    for arm in ["right", "left"]:
        if not robot.set_tool_flange_output_voltage(arm, 12):
            logging.error(f"Failed to set tool flange output voltage ({arm}) as 12v")
            exit(1)
    robot.set_parameter("joint_position_command.cutoff_frequency", "3")
    
    print(f"move to {MANIPULATION_POSE_DICT[position_idx]}")
    move_j_safe(robot, pose=MANIPULATION_POSE[MANIPULATION_POSE_DICT[position_idx]],base_cmd=base_cmd, safety_dict=safety_dict, model=model, minimum_time=5)

    def robot_state_callback(state: rby.RobotState_A):
        nonlocal robot_q, robot_state
        robot_q = state.position
        robot_state = state
        
    robot.start_state_update(robot_state_callback, 1 / Settings.master_arm_loop_period)

    # ===== SETUP GRIPPER =====
    gripper = Gripper()
    if not gripper.initialize():
        logging.error("Failed to initialize gripper")
        robot.stop_state_update()
        robot.power_off("12v")
        exit(1)
    gripper.homing()
    gripper.start()
    
    # ===== SETUP MASTER ARM =====
    rby.upc.initialize_device(rby.upc.MasterArmDeviceName)
    master_arm_model = f"{os.path.dirname(os.path.realpath(__file__))}/../models/master_arm/model.urdf"
    master_arm = rby.upc.MasterArm(rby.upc.MasterArmDeviceName)
    master_arm.set_model_path(master_arm_model)
    master_arm.set_control_period(Settings.master_arm_loop_period)
    active_ids = master_arm.initialize(verbose=False)
    if len(active_ids) != rby.upc.MasterArm.DeviceCount:
        logging.error(
            f"Mismatch in the number of devices detected for RBY Master Arm (active devices: {active_ids})"
        )
        exit(1)

    right_q = None
    left_q = None
    right_minimum_time = 1.0
    left_minimum_time = 1.0
    stream = robot.create_command_stream(priority=1)
    should_align = True
    
    def master_arm_control_loop(state: rby.upc.MasterArm.State):
        nonlocal position_mode, right_q, left_q, right_minimum_time, left_minimum_time, should_align, robot_state
                    
        gripper.set_target(
            np.array(
                [state.button_right.trigger / 1000, state.button_left.trigger / 1000]
            )
        )
        ma_input = rby.upc.MasterArm.ControlInput()
        if should_align:
            ma_input.target_operating_mode[0:14].fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
            ma_input.target_torque[0:14].fill(5)
            
            # for smooth alignment
            ma_input.target_position[0:7] = limit_joint_movement(
                state.q_joint[0:7], 
                robot_state.position[model.right_arm_idx].tolist()
            )
            ma_input.target_position[7:14] = limit_joint_movement(
                state.q_joint[7:14], 
                robot_state.position[model.left_arm_idx].tolist()
            )
            
            right_q = ma_input.target_position[0:7]
            left_q = ma_input.target_position[7:14]
        
        else:
            # ===== CALCULATE MASTER ARM COMMAND =====
            torque = (
                state.gravity_term
                + Settings.master_arm_q_limit_barrier
                * (
                    np.maximum(Settings.master_arm_min_q - state.q_joint, 0)
                    + np.minimum(Settings.master_arm_max_q - state.q_joint, 0)
                )
                + Settings.master_arm_viscous_gain * state.qvel_joint
            )
            torque = np.clip(torque, -Settings.master_arm_torque_limit, Settings.master_arm_torque_limit)
            if state.button_right.button == 1:      # equals 1 means teleoperation mode
                ma_input.target_operating_mode[0:7].fill(
                    rby.DynamixelBus.CurrentControlMode
                )
                ma_input.target_torque[0:7] = torque[0:7]
                right_q = state.q_joint[0:7]
            else:
                ma_input.target_operating_mode[0:7].fill(
                    rby.DynamixelBus.CurrentBasedPositionControlMode
                )
                ma_input.target_torque[0:7].fill(5)
                ma_input.target_position[0:7] = right_q

            if state.button_left.button == 1:
                ma_input.target_operating_mode[7:14].fill(
                    rby.DynamixelBus.CurrentControlMode
                )
                ma_input.target_torque[7:14] = torque[7:14]
                left_q = state.q_joint[7:14]
            else:
                ma_input.target_operating_mode[7:14].fill(
                    rby.DynamixelBus.CurrentBasedPositionControlMode
                )
                ma_input.target_torque[7:14].fill(5)
                ma_input.target_position[7:14] = left_q

            # Check whether target configure is in collision
            q = robot_q.copy()
            q[model.right_arm_idx] = right_q
            q[model.left_arm_idx] = left_q
            dyn_state.set_q(q)
            dyn_model.compute_forward_kinematics(dyn_state)
            
            collision_result = dyn_model.detect_collisions_or_nearest_links(dyn_state, 1)[0]
            distance = collision_result.distance
            is_collision = distance < 0.02
            
            if is_collision:
                logging.error("Collision detected - Command aborted")
                return ma_input

            # ===== BUILD ROBOT COMMAND USING SAFETY CONSTRAINTS =====
            rc = rby.BodyComponentBasedCommandBuilder()
            if state.button_right.button and not is_collision:
                right_minimum_time -= Settings.master_arm_loop_period
                right_minimum_time = max(
                    right_minimum_time, Settings.master_arm_loop_period * 1.01
                )
                
                # Use safety constraints from safety_dict
                right_arm_position = np.clip(
                    right_q,
                    safety_dict["robot_min_q"][model.right_arm_idx],
                    safety_dict["robot_max_q"][model.right_arm_idx],
                )
                
                right_arm_builder = (
                    rby.JointPositionCommandBuilder()
                    if position_mode
                    else rby.JointImpedanceControlCommandBuilder()
                )
                (
                    right_arm_builder.set_command_header(
                        rby.CommandHeaderBuilder().set_control_hold_time(1e6)
                    )
                    .set_position(right_arm_position)
                    .set_velocity_limit(safety_dict["robot_max_qdot"][model.right_arm_idx])
                    .set_acceleration_limit(safety_dict["robot_max_qddot"][model.right_arm_idx] * 30)
                    .set_minimum_time(right_minimum_time)
                )
                if not position_mode:
                    (
                        right_arm_builder.set_stiffness(
                            [Settings.impedance_stiffness] * len(model.right_arm_idx)
                        )
                        .set_damping_ratio(Settings.impedance_damping_ratio)
                        .set_torque_limit(
                            [Settings.impedance_torque_limit] * len(model.right_arm_idx)
                        )
                    )
                rc.set_right_arm_command(right_arm_builder)
            else:
                right_minimum_time = 0.8

            if state.button_left.button and not is_collision:
                left_minimum_time -= Settings.master_arm_loop_period
                left_minimum_time = max(
                    left_minimum_time, Settings.master_arm_loop_period * 1.01
                )
                
                # Use safety constraints from safety_dict
                left_arm_position = np.clip(
                    left_q,
                    safety_dict["robot_min_q"][model.left_arm_idx],
                    safety_dict["robot_max_q"][model.left_arm_idx],
                )
                
                left_arm_builder = (
                    rby.JointPositionCommandBuilder()
                    if position_mode
                    else rby.JointImpedanceControlCommandBuilder()
                )
                (
                    left_arm_builder.set_command_header(
                        rby.CommandHeaderBuilder().set_control_hold_time(1e6)
                    )
                    .set_position(left_arm_position)
                    .set_velocity_limit(safety_dict["robot_max_qdot"][model.left_arm_idx])
                    .set_acceleration_limit(safety_dict["robot_max_qddot"][model.left_arm_idx] * 30)
                    .set_minimum_time(left_minimum_time)
                )
                if not position_mode:
                    (
                        left_arm_builder.set_stiffness(
                            [Settings.impedance_stiffness] * len(model.left_arm_idx)
                        )
                        .set_damping_ratio(Settings.impedance_damping_ratio)
                        .set_torque_limit(
                            [Settings.impedance_torque_limit] * len(model.left_arm_idx)
                        )
                    )
                rc.set_left_arm_command(left_arm_builder)
            else:
                left_minimum_time = 0.8
            
            stream.send_command(
                rby.RobotCommandBuilder().set_command(
                    rby.ComponentBasedCommandBuilder().set_body_command(rc)
                )
            )

        return ma_input

    # ===== SETUP ROS =====
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    
    rclpy.init()
    node = rclpy.create_node("teleoperation")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    
    # Create a thread to spin the executor
    def ros_spin_thread():
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.001)
    
    # Start the ROS spinning thread
    spin_thread = threading.Thread(target=ros_spin_thread, daemon=True)
    spin_thread.start()
    
    rate = node.create_rate(Settings.data_collection_rate)
    
    # pub robot state
    state_pub = node.create_publisher(RobotStateMsg, "/robot/state", 10)
    
    # sub episode done signal
    def episode_done_callback(msg: Bool):
        nonlocal episode_done
        with episode_lock:
            episode_done = msg.data
        print(f"Received episode_done signal: {msg.data}")
    node.create_subscription(Bool, "/collection/done", episode_done_callback, 1)
    
    # sub episode start signal
    def episode_start_callback(msg: Bool):
        nonlocal episode_start
        with episode_lock:
            episode_start = msg.data
        print(f"Received episode_start signal: {msg.data}")
    node.create_subscription(Bool, "/collection/start", episode_start_callback, 1)

    # sub base control
    def BaseControl_cmd(msg: Twist):
        nonlocal last_base_time, base_cmd
        last_base_time = time.time()
        base_cmd = BaseCmdData(msg.linear.x, msg.angular.z)
    node.create_subscription(Twist, "/cmd_vel", BaseControl_cmd, 1)

    # ===== SETUP SIGNAL =====
    def handler(signum, frame):
        robot.stop_state_update()
        master_arm.stop_control()
        robot.cancel_control()
        
        # Stop ROS executor and shutdown
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

        time.sleep(0.5)

        robot.disable_control_manager()
        robot.power_off("12v")
        gripper.stop()
        exit(1)

    signal.signal(signal.SIGINT, handler)
    
    # ===== Episodic LOOP =====
    master_arm.start_control(master_arm_control_loop)
    gripper.start_motor_states_update()
    while True:
        # ===== ALIGN =====
        # this is for 3rd position (right arm intermediate)
        if position_idx == 2:
            stream.send_command(
            joint_position_command_builder_safe(
                pose=MANIPULATION_POSE["shelf_bottom_intermediate"],
                safety_dict=safety_dict,
                model=model,
                minimum_time=3.0,
                control_hold_time=1e6,
                base_cmd=base_cmd,
                )
            )
            time.sleep(3.0)
            
        stream.send_command(
            joint_position_command_builder_safe(
                pose=MANIPULATION_POSE[MANIPULATION_POSE_DICT[position_idx]],
                safety_dict=safety_dict,
                model=model,
                minimum_time=3.0,
                control_hold_time=1e6,
                base_cmd=base_cmd,
            )
        )
        time.sleep(3.0)
        
        # ===== ROS LOOP =====
        # Reset episode flags before starting
        with episode_lock:
            episode_start = False
            episode_done = False
        
        wait_count = 0
        print("Waiting for episode start signal...", flush=True)
        while True:
            with episode_lock:
                if episode_start:
                    break
            
            if base_cmd is not None:
                # ===== SET BASE CMD BUT LAZY STYLE =====
                if last_base_time is not None:
                    time_diff = time.time() - last_base_time
                    if time_diff > 0.1:     #0.1s
                        base_cmd = BaseCmdData(0.0, 0.0)
                
                # ====== UPDATE WBC CMD ======
                wbc_command = joint_position_command_builder_safe(
                    pose=MANIPULATION_POSE[MANIPULATION_POSE_DICT[position_idx]],
                    safety_dict=safety_dict,
                    base_cmd=base_cmd,
                    minimum_time=0.1,
                    model=model,
                    control_hold_time=1e6,
                )
                stream.send_command(wbc_command)
                time.sleep(1.0/Settings.rby1_control_frequency+0.1)
            else:
                wait_count += 1
                if wait_count % 10 == 0:  # Print status every ~1 second
                    print(f"Waiting for episode start... ({wait_count/10} sec)", flush=True)
                time.sleep(0.1)
        
        should_align = False
        
        print("Episode started! Running teleoperation...", flush=True)
        loop_counter = 0
        
        while True:
            loop_counter += 1
            
            # Check if episode is done
            with episode_lock:
                if episode_done:
                    print("Episode complete. Resetting position...")
                    should_align = True
                    break
            
            # Print status update occasionally
            if loop_counter % 1000 == 0:
                print(f"Teleoperation running... (loop {loop_counter})", flush=True)
            
            # Get gripper data
            gripper_motor_states = gripper.get_motor_states()
            gripper_target_qpos = gripper.target_q

            # Publish robot state
            if gripper_motor_states is not None and robot_state is not None and gripper_target_qpos is not None and gripper.min_q is not None and gripper.max_q is not None:
                try:
                    # Create gripper state dictionary
                    gripper_state = {
                        "left": {
                            "qpos": gripper_motor_states[1][1],
                            "target_qpos": gripper_target_qpos[1],
                            "min_qpos": gripper.min_q[1],
                            "max_qpos": gripper.max_q[1],
                        },
                        "right": {
                            "qpos": gripper_motor_states[0][1],
                            "target_qpos": gripper_target_qpos[0],
                            "min_qpos": gripper.min_q[0],
                            "max_qpos": gripper.max_q[0],
                        },
                    }
                    
                    # Create and publish state message
                    state_msg = create_robot_state_msg(robot_state, model, gripper_state)
                    state_pub.publish(state_msg)

                except Exception as e:
                    print(f"Error publishing state: {e}", flush=True)

            # Sleep to maintain the desired rate
            rate.sleep()

        print("Preparing for next episode...", flush=True)
        time.sleep(1.0)  # wait for final commands to finish

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="teleoperation")
    parser.add_argument("--address", type=str, default="192.168.30.1:50051", help="Robot address")
    parser.add_argument(
        "--model", type=str, default="a", help="Robot Model Name (default: 'a')"
    )
    parser.add_argument(
        "--power",
        type=str,
        default=".*",
        help="Regex pattern for power device names (default: '.*')",
    )
    parser.add_argument(
        "--servo",
        type=str,
        default="torso_.*|right_arm_.*|left_arm_.*",
        help="Regex pattern for servo names (default: 'torso_.*|right_arm_.*|left_arm_.*')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="position",
        choices=["position", "impedance"],
        help="Control mode to use: 'position' or 'impedance' (default: 'position')",
    )
    args = parser.parse_args()

    main(args.address, args.model, args.power, args.servo, args.mode)