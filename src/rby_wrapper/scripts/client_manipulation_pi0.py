#!/usr/bin/env python3
# subscribe the robotState and image
# instantiate the manipulation policy
# publish the action

import queue
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rby_wrapper.msg import RobotState
from rby_wrapper.msg import RobotControl
import cv2
import numpy as np
import logging
import time
import os
import sys
from std_msgs.msg import Bool
import openpi_client
from collections import deque

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

right_gripper_max_qpos = 6.14052382
right_gripper_min_qpos = -2.67679592

class DataBuffer():
    def __init__(self):
        self.right_arm_qpos = None
        self.right_gripper_qpos_normalized = None
        self.head_img = None
        self.right_img = None
        
    def get_obs(self):
        return self.robot_state, self.image

def convert_image(msg: CompressedImage) -> np.ndarray:
    """Convert ROS CompressedImage message to numpy array"""
    try:
        # Convert CompressedImage message to numpy array
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        logging.error(f'Error converting compressed image: {str(e)}')
        return None

class ManipulationPolicy():
    def __init__(self, node: Node):
        self.node = node
        # ===== model =====
        self.policy = openpi_client.WebsocketClientPolicy()

        # ===== data buffer =====
        self.data_buffer = DataBuffer()
        self.policy_active = False
        self.action_chunk = None    # define it as a cueue, to receive the policy output
        
        # ===== subscribe the robotState and image =====
        self.robot_state_sub = self.node.create_subscription(RobotState, '/robot/state', self.robot_state_callback, 1)
        self.right_image_sub = self.node.create_subscription(CompressedImage, '/right_camera/right_camera/color/image_rect_raw/compressed', self.right_image_callback, 1)
        self.head_image_sub = self.node.create_subscription(CompressedImage, '/head_camera/head_camera/color/image_raw/compressed', self.head_image_callback, 1)
        self.policy_status_sub = self.node.create_subscription(Bool, '/manager/call_policy', self.policy_status_callback, 1)
        
        # ===== publish the action =====
        self.robot_control_pub = self.node.create_publisher(RobotControl, '/robot/control', 1)
        self.inference_timer = self.node.create_timer(1/15, self.inference_callback)
    
    def setup_policy(self):
        pass
    
    def robot_state_callback(self, msg: RobotState):
        self.data_buffer.right_arm_qpos = msg.right_arm_qpos
        self.data_buffer.right_gripper_qpos = msg.right_gripper_qpos
        self.data_buffer.right_gripper_qpos_normalized = msg.right_gripper_qpos_normalized

    def right_image_callback(self, msg: CompressedImage):
        self.data_buffer.right_img = convert_image(msg)

    def head_image_callback(self, msg: CompressedImage):
        self.data_buffer.head_img = convert_image(msg)

    def policy_status_callback(self, msg: Bool):
        self.policy_active = msg.data
        logging.info(f'Policy status updated: {"Active" if self.policy_active else "Inactive"}')

    def inference_callback(self):
        if not hasattr(self, 'action_chunk') or not self.action_chunk:
            if (self.data_buffer.right_img is not None and
                self.data_buffer.head_img is not None and
                self.data_buffer.right_gripper_qpos_normalized is not None):
                time_start = time.time()

                print(f"right_gripper_qpos_normalized: {self.data_buffer.right_gripper_qpos_normalized}")
                obs = {
                    "observation/right_joint_position": np.array(self.data_buffer.right_arm_qpos),
                    "observation/right_gripper_position": np.array([self.data_buffer.right_gripper_qpos_normalized]),
                    "observation/head_camera": self.data_buffer.head_img,
                    "observation/right_wrist_camera": self.data_buffer.right_img,
                    "prompt": "lamp"  # not sure if this is correct
                }
                # obs = {
                #     "observation/right_joint_position": np.array(self.data_buffer.right_arm_qpos),
                #     "observation/right_gripper_position": np.array([0.5]),
                #     "observation/head_camera": self.data_buffer.head_img,
                #     "observation/right_wrist_camera": self.data_buffer.right_img,
                #     "prompt": "lamdfdfdfdfsdfasdf"  # not sure if this is correct
                # }
                # print("self.data_buffer.right_arm_qpos", self.data_buffer.right_arm_qpos)
                
                # Get actions and convert to deque
                policy_output = self.policy.infer(obs)
                actions = policy_output['actions']  # shape: (N, 8)
                # print("actions", actions)
                self.action_chunk = deque(actions)
                print(f"action_chunk: {self.action_chunk}")
                time_end = time.time()
                logging.info(f'Inference time: {time_end - time_start} seconds')
        if self.action_chunk is None:
            return
        print(len(self.action_chunk))
        action = self.action_chunk.popleft() if self.action_chunk else None
        print("executing action", action)
        if not self.policy_active:
            return
    
        if action is not None:
            logging.info(f'Action: {action}')
            control_msg = RobotControl()
            control_msg.command_type = 'right_arm'
            control_msg.right_arm_qpos = action[:7].tolist()
            control_msg.right_gripper_qpos_normalized = float(1-action[7])
            
            self.robot_control_pub.publish(control_msg)

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('manipulation_policy')
    policy = ManipulationPolicy(node)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
