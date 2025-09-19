#!/usr/bin/env python3

import time
import argparse
import rby1_sdk as rby
import numpy as np
import rclpy
from rclpy.node import Node
from rby_wrapper.msg import RobotState as RobotStateMsg
from rby_wrapper.msg import RobotControl as RobotControlMsg
from builtin_interfaces.msg import Time
from util_msg import create_robot_state_msg
from util_components import Gripper
from util_command import WBCCommand, BaseCmdData
from util_pose import MANIPULATION_POSE_DICT, MANIPULATION_POSE, Pose, SE2_to_xytheta
import logging
from util_settings import Settings
from util_command import joint_position_command_builder_safe, move_j_safe
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
import threading
from tf2_ros import TransformBroadcaster
import geometry_msgs.msg
from std_msgs.msg import Empty

"""
RBY1 Robot Server

This script provides an interface to control the RBY1 robot using ROS2.
Features include:
- Arm, torso, and gripper position control
- Base movement control using cmd_vel messages
- Collision detection and safety limits
- Robot state publishing
"""

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class RBY1_ROBOT():
    
    # ===== CALLBACK LIST=====
    def robot_state_callback(self):
        self.robot_state = self.robot.get_state()
        self.robot_q = self.robot_state.position
        self.robot_left_arm_qpos = self.robot_q[self.model.left_arm_idx]
        self.robot_right_arm_qpos = self.robot_q[self.model.right_arm_idx]
        self.robot_torso_qpos = self.robot_q[self.model.torso_idx]
        
    def publish_robot_state_callback(self):
        gripper_motor_states = self.gripper.get_motor_states()
        gripper_target_qpos = self.gripper.target_q
        
        # Publish robot state
        if self.robot_state is not None:

            try:
                # Publish TF transform from odom to base
                x, y, theta = SE2_to_xytheta(self.robot_state.odometry)
                self.publish_odom_to_base_tf(x, y, theta)
                
                # Publish joint transforms for torso and arms
                joint_state = self.robot_state.position
                self.publish_joint_transforms(joint_state)
                
                # Publish odometry message
                self.publish_odometry(x, y, theta)
                
                # Create gripper state dictionary
                gripper_state = {
                    "left": {
                        "qpos": gripper_motor_states[1][1] if gripper_motor_states is not None else 0.0,
                        "target_qpos": gripper_target_qpos[1] if gripper_target_qpos is not None else 0.0,
                        "min_qpos": self.gripper.min_q[1],
                        "max_qpos": self.gripper.max_q[1],
                    },
                    "right": {
                        "qpos": gripper_motor_states[0][1] if gripper_motor_states is not None else 0.0,
                        "target_qpos": gripper_target_qpos[0] if gripper_target_qpos is not None else 0.0,
                        "min_qpos": self.gripper.min_q[0],
                        "max_qpos": self.gripper.max_q[0],
                    },
                }
                
                # Create and publish state message
                state_msg = create_robot_state_msg(self.robot_state, self.model, gripper_state)
                self.pub_robot_state.publish(state_msg)
                
            except Exception as e:
                print(f"Error publishing state: {e}", flush=True)
                
    def reset_callback(self, msg: Empty):
        print("reset_callback at", time.time(), flush=True)
        self.last_last_stream_reset_time = self.last_stream_reset_time
        self.last_stream_reset_time = time.time()

    def publish_odom_to_base_tf(self, x, y, theta):
        """
        Publish TF transform from odom to base frame
        """
        # Create transform message
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base"
        
        # Set translation
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        
        # Set rotation (convert from theta to quaternion)
        from math import sin, cos
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = sin(theta/2.0)
        t.transform.rotation.w = cos(theta/2.0)
        
        # Publish the transform
        self.tf_broadcaster.sendTransform(t)

    def publish_odometry(self, x, y, theta):
        """
        Publish odometry message with the robot's position
        """
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.node.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base"
        
        # Set position
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = 0.0
        
        # Set orientation as quaternion
        from math import sin, cos
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = sin(theta/2.0)
        odom_msg.pose.pose.orientation.w = cos(theta/2.0)
        
        # Set velocity (if available from robot state)
        if hasattr(self.robot_state, 'velocity') and len(self.robot_state.velocity) >= 2:
            # Linear velocity
            base_vel = self.robot_state.velocity[0]  # Assuming first element is linear velocity
            odom_msg.twist.twist.linear.x = base_vel
            
            # Angular velocity
            base_omega = self.robot_state.velocity[1]  # Assuming second element is angular velocity
            odom_msg.twist.twist.angular.z = base_omega
        
        # Publish the odometry message
        self.pub_odom.publish(odom_msg)

    def publish_joint_transforms(self, joint_state):
        """
        Publish TF transforms for torso and arm joints based on joint positions and URDF info
        """
        # Get joint indices for each group
        torso_idx = self.model.torso_idx
        left_arm_idx = self.model.left_arm_idx
        right_arm_idx = self.model.right_arm_idx

        # Get current time for all transforms
        current_time = self.node.get_clock().now().to_msg()
        
        # Define rotation axes for each joint based on URDF
        # From URDF, we know the axis of rotation for each joint
        torso_axes = [
            [0, 0, 1],  # joint_torso_0: z-axis
            [0, 1, 0],  # joint_torso_1: y-axis
            [0, 1, 0],  # joint_torso_2: y-axis
            [0, 1, 0],  # joint_torso_3: x-axis
            [0, 0, 1],  # joint_torso_4: z-axis
            [0, 0, 1],  # joint_torso_5: z-axis 
        ]
        
        left_arm_axes = [
            [0, 1, 0],  # left_arm_0: y-axis
            [1, 0, 0],  # left_arm_1: x-axis
            [0, 0, 1],  # left_arm_2: z-axis
            [0, 1, 0],  # left_arm_3: y-axis
            [0, 0, 1],  # left_arm_4: z-axis
            [0, 1, 0],  # left_arm_5: y-axis
            [0, 0, 1],  # left_arm_6: z-axis
        ]
        
        right_arm_axes = [
            [0, 1, 0],  # right_arm_0: y-axis
            [1, 0, 0],  # right_arm_1: x-axis
            [0, 0, 1],  # right_arm_2: z-axis
            [0, 1, 0],  # right_arm_3: y-axis
            [0, 0, 1],  # right_arm_4: z-axis
            [0, 1, 0],  # right_arm_5: y-axis
            [0, 0, 1],  # right_arm_6: z-axis
        ]
        
        # Define joint origins (simplified, should ideally come from URDF)
        # These are approximate values from the URDF
        torso_origins = [
            [0.0, 0.0, 0.2805],  # base to torso_0
            [0.0, 0.0, 0.0],     # torso_0 to torso_1
            [0.0, 0.0, 0.3],     # torso_1 to torso_2
            [0.0, 0.0, 0.35],    # torso_2 to torso_3
            [0.0, 0.0, 0.0],     # torso_3 to torso_4
            [0.0, 0.0, 0.309],    # torso_4 to torso_5
        ]
        
        # Create transforms for torso joints
        for i, idx in enumerate(torso_idx):
            t = TransformStamped()
            t.header.stamp = current_time
            
            # Set parent and child frame IDs
            if i == 0:
                t.header.frame_id = "base"
                t.child_frame_id = f"link_torso_{i}"
            else:
                t.header.frame_id = f"link_torso_{i-1}"
                t.child_frame_id = f"link_torso_{i}"
            
            # Get joint angle
            joint_angle = joint_state[idx]
            
            # Set translation from URDF info
            if i < len(torso_origins):
                t.transform.translation.x = torso_origins[i][0]
                t.transform.translation.y = torso_origins[i][1]
                t.transform.translation.z = torso_origins[i][2]
            else:
                # Default translation if index is out of range
                t.transform.translation.x = 0.0
                t.transform.translation.y = 0.0
                t.transform.translation.z = 0.0
            
            # Set rotation based on joint angle and rotation axis
            from math import sin, cos
            # Ensure we don't go out of bounds
            axis_idx = min(i, len(torso_axes) - 1)
            axis = torso_axes[axis_idx]
            half_angle = joint_angle / 2.0
            
            # Create quaternion for the rotation
            if axis[0] == 1:  # x-axis rotation
                t.transform.rotation.x = sin(half_angle)
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = cos(half_angle)
            elif axis[1] == 1:  # y-axis rotation
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = sin(half_angle)
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = cos(half_angle)
            elif axis[2] == 1:  # z-axis rotation
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = sin(half_angle)
                t.transform.rotation.w = cos(half_angle)
            
            # Send the transform
            self.tf_broadcaster.sendTransform(t)
        
        # Create transforms for left arm joints
        # Left arm origin connects to the last torso link
        left_arm_origins = [
            [0.0, 0.22, 0.08],     # torso_5 to left_arm_0
            [0.0, 0.0, 0.0],      # left_arm_0 to left_arm_1
            [0.0, 0.0, 0.0],      # left_arm_1 to left_arm_2
            [0.031, 0.0, -0.276], # left_arm_2 to left_arm_3
            [-0.031, 0.0, -0.256], # left_arm_3 to left_arm_4
            [0.0, 0.0, 0.0],      # left_arm_4 to left_arm_5
            [0.0, 0.0, 0.0],      # left_arm_5 to left_arm_6
        ]
        
        for i, idx in enumerate(left_arm_idx):
            t = TransformStamped()
            t.header.stamp = current_time
            
            if i == 0:
                t.header.frame_id = "link_torso_5"  # Arm connects to the last torso link
                t.child_frame_id = f"link_left_arm_{i}"
            else:
                t.header.frame_id = f"link_left_arm_{i-1}"
                t.child_frame_id = f"link_left_arm_{i}"
            
            # Get joint angle
            joint_angle = joint_state[idx]
            
            # Set translation from URDF info
            if i < len(left_arm_origins):
                t.transform.translation.x = left_arm_origins[i][0]
                t.transform.translation.y = left_arm_origins[i][1]
                t.transform.translation.z = left_arm_origins[i][2]
            else:
                # Default translation if index is out of range
                t.transform.translation.x = 0.0
                t.transform.translation.y = 0.0
                t.transform.translation.z = 0.0
            
            # Set rotation based on joint angle and rotation axis
            from math import sin, cos
            # Ensure we don't go out of bounds
            axis_idx = min(i, len(left_arm_axes) - 1)
            axis = left_arm_axes[axis_idx]
            half_angle = joint_angle / 2.0
            
            # Create quaternion for the rotation
            if axis[0] == 1:  # x-axis rotation
                t.transform.rotation.x = sin(half_angle)
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = cos(half_angle)
            elif axis[1] == 1:  # y-axis rotation
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = sin(half_angle)
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = cos(half_angle)
            elif axis[2] == 1:  # z-axis rotation
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = sin(half_angle)
                t.transform.rotation.w = cos(half_angle)
            
            # Send the transform
            self.tf_broadcaster.sendTransform(t)
        
        # Create transforms for right arm joints
        # Right arm origin connects to the last torso link
        right_arm_origins = [
            [0.0, -0.22, 0.08],    # torso_5 to right_arm_0
            [0.0, 0.0, 0.0],      # right_arm_0 to right_arm_1
            [0.0, 0.0, 0.0],      # right_arm_1 to right_arm_2
            [0.031, 0.0, -0.276], # right_arm_2 to right_arm_3
            [-0.031, 0.0, -0.256], # right_arm_3 to right_arm_4
            [0.0, 0.0, 0.0],      # right_arm_4 to right_arm_5
            [0.0, 0.0, 0.0],      # right_arm_5 to right_arm_6
        ]
        
        for i, idx in enumerate(right_arm_idx):
            t = TransformStamped()
            t.header.stamp = current_time
            
            if i == 0:
                t.header.frame_id = "link_torso_5"  # Arm connects to the last torso link
                t.child_frame_id = f"link_right_arm_{i}"
            else:
                t.header.frame_id = f"link_right_arm_{i-1}"
                t.child_frame_id = f"link_right_arm_{i}"
            
            # Get joint angle
            joint_angle = joint_state[idx]
            
            # Set translation from URDF info
            if i < len(right_arm_origins):
                t.transform.translation.x = right_arm_origins[i][0]
                t.transform.translation.y = right_arm_origins[i][1]
                t.transform.translation.z = right_arm_origins[i][2]
            else:
                # Default translation if index is out of range
                t.transform.translation.x = 0.0
                t.transform.translation.y = 0.0
                t.transform.translation.z = 0.0
            
            # Set rotation based on joint angle and rotation axis
            from math import sin, cos
            # Ensure we don't go out of bounds
            axis_idx = min(i, len(right_arm_axes) - 1)
            axis = right_arm_axes[axis_idx]
            half_angle = joint_angle / 2.0
            
            # Create quaternion for the rotation
            if axis[0] == 1:  # x-axis rotation
                t.transform.rotation.x = sin(half_angle)
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = cos(half_angle)
            elif axis[1] == 1:  # y-axis rotation
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = sin(half_angle)
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = cos(half_angle)
            elif axis[2] == 1:  # z-axis rotation
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = sin(half_angle)
                t.transform.rotation.w = cos(half_angle)
            
            # Send the transform
            self.tf_broadcaster.sendTransform(t)

    def update_command_callback(self, command: RobotControlMsg):
        if command.command_type == "base":
            self.last_base_time = time.time()
            self.command.base_cmd = BaseCmdData(command.base_vel, command.base_omega)
        elif command.command_type == "right_arm":
            self.command.right_arm_qpos = command.right_arm_qpos
            self.command.right_gripper_qpos_normalized = command.right_gripper_qpos_normalized
        elif command.command_type == "torso":
            self.command.torso_qpos = command.torso_qpos
        elif command.command_type == "gripper":
            self.command.right_gripper_qpos_normalized = command.right_gripper_qpos_normalized
        else:
            logging.error(f"Invalid command type: {command.command_type}")

    def cmd_vel_callback(self, msg):
        self.last_base_time = time.time()
        self.command.base_cmd = BaseCmdData(msg.linear.x, msg.angular.z)

    def wbc_callback(self):
        # ===== command check =====
        # gripper width check
        self.command.right_gripper_qpos_normalized = np.clip(self.command.right_gripper_qpos_normalized, 0.0, 1.0)
        self.command.left_gripper_qpos_normalized = np.clip(self.command.left_gripper_qpos_normalized, 0.0, 1.0)
        
        # collision check
        q = self.robot_q.copy()
        q[self.model.right_arm_idx] = self.command.right_arm_qpos
        q[self.model.left_arm_idx] = self.command.left_arm_qpos
        self.dyn_state.set_q(q)
        self.dyn_model.compute_forward_kinematics(self.dyn_state)
        
        collision_result = self.dyn_model.detect_collisions_or_nearest_links(self.dyn_state, 1)[0]
        distance = collision_result.distance
        is_collision = distance < 0.02
        
        if is_collision:
            logging.error("Collision detected - Command aborted")
            return
        
        # ====== SET BODY CMD ======
        pose = Pose(
            right_arm=self.command.right_arm_qpos,
            left_arm=self.command.left_arm_qpos,
            torso=self.command.torso_qpos,
        )
        
        # ===== SET BASE CMD BUT LAZY STYLE =====
        if self.last_base_time is not None:
            time_diff = time.time() - self.last_base_time
            if time_diff > self.base_timeout_threshold:
                self.command.base_cmd = BaseCmdData(0.0, 0.0)
        
        # ====== UPDATE WBC CMD ======
        wbc_command = joint_position_command_builder_safe(
            pose=pose,
            safety_dict=self.safety_dict,
            base_cmd=self.command.base_cmd,
            minimum_time=1/Settings.rby1_control_frequency,
            model=self.model,
            control_hold_time=1e6,
        )
        
        # ===== RESET STREAM IF NEEDED ======
        if self.last_last_stream_reset_time is not None:
            diff = self.last_stream_reset_time - self.last_last_stream_reset_time
            if diff < 0.5:
                logging.info("Resetting robot controller")
                self.stream_list[-1].cancel()
                while not self.stream_list[-1].is_done():
                    print("waiting for stream to be done", flush=True)
                    time.sleep(0.001)
                self.stream_list.append(self.robot.create_command_stream(priority=1))
                time.sleep(0.1) # wait for stream to be created
                logging.info("Resetting robot controller done.")
                self.last_last_stream_reset_time = None

        # ===== SEND WBC CMD ======
        if wbc_command is not None:
            self.stream_list[-1].send_command(wbc_command)

        # ====== SET GRIPPER CMD ======
        self.gripper.set_target(np.array([self.command.right_gripper_qpos_normalized, self.command.left_gripper_qpos_normalized]))
    
    # ===== INITIALIZATION =====
    def __init__(self,
                 node: Node,
                 address: str = Settings.rby1_address,
                 model: str = Settings.rby1_model,
                 power: str = Settings.rby1_power,
                 servo: str = Settings.rby1_servo,
                 control_mode: str=Settings.rby1_control_mode):
        
        # ===== CHECK ARGUMENTS =====
        self.node = node
        supported_model = ["A"]
        supported_control_mode = ["position"]
        if not model in supported_model:
            logging.error(f"Model {model} not supported (Current supported model is {supported_model})")
            exit(1)
        if not control_mode in supported_control_mode:
            logging.error(f"Control mode {control_mode} not supported (Current supported control mode is {supported_control_mode})")
            exit(1)
        
        # ===== CHOOSE MANIPULATION POSE =====
        print("Choose position index:")
        for key, item in MANIPULATION_POSE_DICT.items():
            print(f"{key}: {item}")
        self.manipulation_pose_idx = int(input("Enter position index:"))
        self.manipulation_pose = MANIPULATION_POSE[MANIPULATION_POSE_DICT[self.manipulation_pose_idx]]

        # ===== SETUP ROBOT =====
        self.robot = rby.create_robot(address, model)
        if not self.robot.connect():
            logging.error(f"Failed to connect robot {address}")
            exit(1)
        self.model = self.robot.model()
        self.dyn_model = self.robot.get_dynamics()
        self.dyn_state = self.dyn_model.make_state([], self.model.robot_joint_names)
        self.robot_q = None
        self.robot_state = None
        self.last_base_time = None
        self.base_timeout_threshold = 0.1   # 0.3s
        
        # ===== SETUP SAFETY DICT =====
        self.robot_max_q = self.dyn_model.get_limit_q_upper(self.dyn_state)
        self.robot_min_q = self.dyn_model.get_limit_q_lower(self.dyn_state)
        self.robot_max_qdot = self.dyn_model.get_limit_qdot_upper(self.dyn_state)
        self.robot_max_qddot = self.dyn_model.get_limit_qddot_upper(self.dyn_state)
        self.robot_max_qdot[self.model.right_arm_idx[-1]] *= 2  # wrist joint can be faster
        self.robot_max_qdot[self.model.left_arm_idx[-1]] *= 2  # wrist joint can be faster
        self.robot_max_qdot[self.model.left_arm_idx] *= 0.25  
        self.robot_max_qdot[self.model.right_arm_idx] *= 0.25
        self.robot_max_qdot[self.model.torso_idx] *= 0.08
        self.robot_max_qddot *= 0.3  # slow down all joints for safety
        
        self.safety_dict = {
            "robot_max_q": self.robot_max_q,
            "robot_min_q": self.robot_min_q,
            "robot_max_qdot": self.robot_max_qdot,
            "robot_max_qddot": self.robot_max_qddot,
            "base_max_acc": 0.3,
            "base_max_vel": 0.5,
            "base_max_omega":0.5
        }
        
        # ===== SETUP TF BROADCASTER =====
        self.tf_broadcaster = TransformBroadcaster(self.node)
        
        # ===== ROBOT INITIALIZATION =====
        if not self.robot.is_power_on(power):
            rv = self.robot.power_on(power)
            if not rv:
                logging.error("Failed to power on")
                raise RuntimeError("Failed to power on robot")
        if not self.robot.is_servo_on(servo):
            if not self.robot.servo_on(servo):
                logging.error(f"Failed to servo ({servo}) on")
                exit(1)

        # Make sure wheel servos are on for base movement
        if not self.robot.is_servo_on(".*(right_wheel|left_wheel).*"):
            if not self.robot.servo_on(".*(right_wheel|left_wheel).*"):
                logging.error("Failed to servo on wheels - Base movement will not work")
                logging.warning("Continuing without wheel control...")
        logging.info(f"Enabled wheel servos, be careful with base movement")
        
        self.robot.reset_fault_control_manager()
        if not self.robot.enable_control_manager():
            logging.error(f"Failed to enable control manager")
            exit(1)
        logging.info(f"Enabled control manager")
        
        for arm in ["right", "left"]:
            if not self.robot.set_tool_flange_output_voltage(arm, 12):
                logging.error(f"Failed to set tool flange output voltage ({arm}) as 12v")
                exit(1)
        self.robot.set_parameter("joint_position_command.cutoff_frequency", "3")
        
        # ===== SETUP GRIPPER =====
        self.gripper = Gripper()
        if not self.gripper.initialize(verbose=True):
            logging.error("Failed to initialize gripper")
            self.robot.disconnect()
            exit(1)
        self.gripper.homing()
        logging.info(f"Calibrated gripper servo")
        
        # ===== SETUP WBC =====
        logging.info(f"Moving to {MANIPULATION_POSE_DICT[self.manipulation_pose_idx]}")
        move_j_safe(self.robot, self.manipulation_pose, self.safety_dict, self.model, base_cmd=BaseCmdData(), minimum_time=0.1) # with safe dict, don't have to set minimum_time
        logging.info("Arrived at the pre-defined pose")
        
        self.command = WBCCommand(self.manipulation_pose)  # should be initialized same as manipulation_pose
        self.stream_list = []
        self.stream_list.append(self.robot.create_command_stream(priority=1))
        self.last_stream_reset_time = None
        self.last_last_stream_reset_time = None
        # ===== reset Odom =====
        self.robot.reset_odometry(0, np.array([[0.0], [0.0]]))
        
        # ===== SETUP TOPIC =====
        self.pub_robot_state = self.node.create_publisher(RobotStateMsg, "/robot/state", 1)
        self.pub_odom = self.node.create_publisher(Odometry, "/robot/odom", 1)
        self.sub_update_command = self.node.create_subscription(RobotControlMsg, "/robot/control", self.update_command_callback, 1)
        self.sub_cmd_vel = self.node.create_subscription(Twist, "/cmd_vel", self.cmd_vel_callback, 1)
        self.sub_reset = self.node.create_subscription(Empty, "/manager/reset_robot", self.reset_callback, 1)
        
        # ===== SETUP TIMER =====
        self.timer_robot_state_update = self.node.create_timer(1/Settings.rby1_state_update_frequency, self.robot_state_callback) # Done
        self.timer_robot_state_pub = self.node.create_timer(1/Settings.rby1_state_update_frequency, self.publish_robot_state_callback) # Done
        self.timer_gripper_state_update = self.node.create_timer(1/Settings.rby1_state_update_frequency, self.gripper.update_motor_states) # Done
        self.timer_gripper_control = self.node.create_timer(1/Settings.rby1_control_frequency, self.gripper.loop) # Done
        self.timer_wbc = self.node.create_timer(1/Settings.rby1_control_frequency, self.wbc_callback)
        
        # ===== SPIN =====
        print("Spinning...")
        rclpy.spin(self.node)


def main(args=None):
    rclpy.init()
    node = rclpy.create_node("rby1_robot")
    RBY1_ROBOT(node)

if __name__ == '__main__':
    main()