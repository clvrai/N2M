#!/usr/bin/env python3
# subscribe the robotState and image
# instantiate the manipulation policy
# publish the action

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rby_wrapper.msg import RobotState
from rby_wrapper.msg import RobotControl
from std_msgs.msg import Int32, Float32MultiArray
from std_msgs.msg import Empty
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from util_tf_broadcaster import publish_odom_to_base_tf, publish_joint_transforms
import tf2_ros
from rby_wrapper.msg import RobotControl
import open3d as o3d
import struct
import yaml
import json
import pathlib
import cv2
import numpy as np
import logging
import time
import os
import sys
from std_msgs.msg import Bool

# map_or_odom
# FRAME_ID = "map"  
FRAME_ID = "odom"  

# Configure yaml for numpy
def numpy_representer(dumper, data):
    """Represent NumPy scalars with full precision."""
    if isinstance(data, np.floating):
        return dumper.represent_float(float(data))
    elif isinstance(data, np.integer):
        return dumper.represent_int(int(data))
    elif isinstance(data, np.ndarray):
        return dumper.represent_list([numpy_representer(dumper, i) for i in data])
    else:
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

yaml.add_representer(np.float64, numpy_representer)
yaml.add_representer(np.float32, numpy_representer)
yaml.add_representer(np.int64, numpy_representer)
yaml.add_representer(np.int32, numpy_representer)
yaml.add_representer(np.ndarray, numpy_representer)

# ===== load model =====
target_path = '/home/mm/workbench/nav2man'
if target_path not in sys.path:
    sys.path.append(target_path)
from nav2man import SIRPredictor
from nav2man.utils.prediction import predict_SIR_target_point, predict_SIR_mean
from nav2man.utils.sample_utils import TargetHelper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define workspace area macros
HALF_X = 0.4
HALF_Y = 0.5
HALF_THETA = np.deg2rad(30)
HALF_TORSO1 = 0.35

# Define target points
object_x = [1.8, 1.8, 1.8, 1.8]
object_y = [0.5, 0.25, -0.25, -0.5]

# Define randomized sampling center
target_x = np.array(object_x) - 0.8
target_y = object_y
target_torso1 = [0.38396602275779107, 7.85394366e-01, 1.1868189130906608]
torso1_min = 0.17
torso1_max = 1.22
# Define current row and column
current_row = 0 # 0, 1, 2 (top to bottom)
current_col = 0 # 0, 1, 2, 3 (left to right)

xmin = target_x[current_col] - HALF_X
xmax = target_x[current_col] + HALF_X
ymin = target_y[current_col] - HALF_Y
ymax = target_y[current_col] + HALF_Y
torso1_constant = target_torso1[current_row]
z_constant = 0.0


class DataBuffer():
    def __init__(self):
        self.pcl_rendered = None    # topic buffer, update in realtime
        self.scene = None   # pcl_rendered_o3d
        self.obs = None  # Store rendered point cloud in Open3D format
        self.current_id = 0  # Store the current experiment ID
        self.item = None  
        # Add SE2 buffer
        self.current_se2 = [0.0, 0.0, 0.0]  # [x, y, theta]
        self.has_valid_se2 = False
        
    def get_obs(self):
        return self.obs
    
    def update_se2(self, x, y, theta):
        self.current_se2 = [x, y, theta]
        self.has_valid_se2 = True

class SIR():
    def __init__(self, node: Node, task: str):
        self.node = node
        self.clock = node.get_clock()
        self.task = task
        
        # ===== model =====
        with open(f'/home/mm/workbench/nav2man/configs/{task}.json', 'r') as f:
            config = json.load(f)
        self.predictor = SIRPredictor(config=config["model"]).to('cuda')
        print("SIR predictor initialized")
        
        # ===== data buffer =====
        self.data_buffer = DataBuffer()
        
        
        # ===== subscribe the robotState and image =====
        self.sir_inference_sub = self.node.create_subscription(Empty, '/manager/call_SIR', self.sir_inference_callback, 1)
        self.sir_execute_sub = self.node.create_subscription(Empty, '/manager/SIR_execute', self.sir_execute_callback, 1)
        self.sample_pose_sub = self.node.create_subscription(Empty, '/manager/sample_pose', self.sample_pose_callback, 1)
        self.reset_sub = self.node.create_subscription(Empty, '/manager/reset', self.reset_callback, 1)
        self.pcl_rendered_sub = self.node.create_subscription(PointCloud2, '/robot/pcl_rendered', self.pcl_rendered_callback, 1)
        self.save_sub = self.node.create_subscription(Bool, '/manager/save_data', self.save_callback, 1)
        self.torso_control_pub = self.node.create_publisher(RobotControl, '/robot/control', 1)
        self.scene_sub = self.node.create_subscription(PointCloud2, '/stitched_pointcloud', self.scene_callback, 1)
        self.row_sub = self.node.create_subscription(Empty, '/manager/row', self.row_callback, 1)
        self.col_sub = self.node.create_subscription(Empty, '/manager/col', self.col_callback, 1)
        
        # Add subscriber for experiment ID
        self.id_sub = self.node.create_subscription(Int32, '/exp/id', self.id_callback, 1)
        
        # Add subscriber for SE2 data
        self.se2_sub = self.node.create_subscription(Float32MultiArray, '/robot/se2',  self.se2_callback, 10)
        self.se2_nav_pub = self.node.create_publisher(Float32MultiArray, '/robot/se2_current_and_target', 1)
        
        # ===== Add rectangle marker publisher =====
        self.marker_pub = self.node.create_publisher(Marker, '/workspace_boundary', 10)
        
        # ===== Timer for creating and publishing rectangle markers =====
        self.marker_timer = self.node.create_timer(0.2, self.publish_workspace_boundary)
        
        # ===== publish the result =====
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)
        self.update_odom_timer = self.node.create_timer(1/30, self.tf_broadcaster_callback)
        self.best_pose = [1.0, 0.0, 0.0, 7.85394366e-01]  # initial, [x, y, theta, torso1]
        self.best_pose_mapFrame = self.best_pose
        
        # Create prediction YAML directory if it doesn't exist
        self.prediction_yaml_path = f'n2m_inference/{task}/prediction.yaml'
        pathlib.Path(f'n2m_inference/{task}').mkdir(parents=True, exist_ok=True)
        
        # Initialize prediction data dictionary
        self.prediction_data = {}
        
        # Load existing prediction data if available
        self.load_prediction_data()
        
        self.node.get_logger().info("SIR node initialized - now using direct SE2 subscription")
    
    def se2_callback(self, msg: Float32MultiArray):
        """
        Callback for receiving SE2 data from C++ node
        """
        if len(msg.data) == 3:
            x, y, theta = msg.data
            self.data_buffer.update_se2(x, y, theta)
            # Log occasionally to avoid flooding
            if not hasattr(self, 'last_se2_log_time') or (time.time() - self.last_se2_log_time) > 5.0:
                self.node.get_logger().debug(f'Updated SE2: x={x:.2f}, y={y:.2f}, theta={theta:.2f}')
                self.last_se2_log_time = time.time()
        else:
            self.node.get_logger().warning(f'Received invalid SE2 data: expected 3 values, got {len(msg.data)}')
            
    def ros_pointcloud2_to_o3d(self, ros_pc2):
        """
        Convert ROS2 PointCloud2 message to Open3D point cloud, using NumPy for optimized performance
        """
        try:
            # Check if point cloud data was received
            if ros_pc2 is None:
                self.node.get_logger().error("Received None point cloud")
                return None
                
            # Parse PointCloud2 message metadata
            width = ros_pc2.width
            height = ros_pc2.height
            point_step = ros_pc2.point_step
            
            # Find field offsets (x, y, z, rgb)
            x_offset = y_offset = z_offset = rgb_offset = None
            for field in ros_pc2.fields:
                if field.name == 'x':
                    x_offset = field.offset
                elif field.name == 'y':
                    y_offset = field.offset
                elif field.name == 'z':
                    z_offset = field.offset
                elif field.name == 'rgb':
                    rgb_offset = field.offset
            
            if x_offset is None or y_offset is None or z_offset is None:
                self.node.get_logger().error("Point cloud missing x/y/z fields")
                return None
            
            # Total number of points
            num_points = width * height
            
            # Directly use NumPy to create structured array
            time0 = time.time()
            
            # Convert byte data to NumPy array
            cloud_data = np.frombuffer(ros_pc2.data, dtype=np.uint8).reshape(num_points, point_step)
            
            # Create empty arrays to store valid points and colors
            xyz = np.zeros((num_points, 3), dtype=np.float32)
            
            # Extract all x, y, z coordinates at once
            xyz[:, 0] = np.frombuffer(cloud_data[:, x_offset:x_offset+4].tobytes(), dtype=np.float32)
            xyz[:, 1] = np.frombuffer(cloud_data[:, y_offset:y_offset+4].tobytes(), dtype=np.float32)
            xyz[:, 2] = np.frombuffer(cloud_data[:, z_offset:z_offset+4].tobytes(), dtype=np.float32)
            
            # Find mask for valid points
            valid_mask = np.isfinite(xyz).all(axis=1)
            xyz_valid = xyz[valid_mask]
            
            # Process colors
            if rgb_offset is not None:
                # Extract all RGB values
                rgb_data = np.frombuffer(cloud_data[:, rgb_offset:rgb_offset+4].tobytes(), dtype=np.uint32)
                
                # Decompose RGB channels
                rgb_array = np.zeros((num_points, 3), dtype=np.float32)
                rgb_array[:, 0] = ((rgb_data >> 16) & 0xFF) / 255.0  # R
                rgb_array[:, 1] = ((rgb_data >> 8) & 0xFF) / 255.0   # G
                rgb_array[:, 2] = (rgb_data & 0xFF) / 255.0          # B
                
                # Keep only colors of valid points
                rgb_valid = rgb_array[valid_mask]
            else:
                # If no color, use white
                rgb_valid = np.ones((xyz_valid.shape[0], 3), dtype=np.float32)
            
            time1 = time.time()
            
            # Create Open3D point cloud
            o3d_pc = o3d.geometry.PointCloud()
            if xyz_valid.size > 0:
                o3d_pc.points = o3d.utility.Vector3dVector(xyz_valid)
                o3d_pc.colors = o3d.utility.Vector3dVector(rgb_valid)
                
                time2 = time.time()
                print("[DEBUG]01:", (time1-time0)*1000, "ms")
                print("[DEBUG]12:", (time2-time1)*1000, "ms")
                
                # Occasionally log point count for monitoring but don't print every time
                if np.random.random() < 0.1:  # About 10% probability to log
                    self.node.get_logger().info(f"Converted point cloud with {xyz_valid.shape[0]} points")
                
                return o3d_pc
            else:
                self.node.get_logger().warning("Point cloud has no valid points")
                return None
        except Exception as e:
            self.node.get_logger().error(f"Error converting ROS2 PointCloud2 to Open3D: {str(e)}")
            return None
    
    # def ros_pointcloud2_to_o3d(self, ros_pc2):
    #     """
    #     Convert ROS2 PointCloud2 message to Open3D point cloud
    #     This implementation doesn't depend on sensor_msgs.point_cloud2 module
    #     """
    #     try:
    #         # Parse PointCloud2 message metadata
    #         width = ros_pc2.width
    #         height = ros_pc2.height
    #         point_step = ros_pc2.point_step
    #         row_step = ros_pc2.row_step
            
    #         # Find field offsets (x, y, z, rgb)
    #         x_offset = y_offset = z_offset = rgb_offset = None
    #         for field in ros_pc2.fields:
    #             if field.name == 'x':
    #                 x_offset = field.offset
    #             elif field.name == 'y':
    #                 y_offset = field.offset
    #             elif field.name == 'z':
    #                 z_offset = field.offset
    #             elif field.name == 'rgb':
    #                 rgb_offset = field.offset
            
    #         if x_offset is None or y_offset is None or z_offset is None:
    #             self.node.get_logger().error("Point cloud missing x/y/z fields")
    #             return None
            
    #         # Create numpy arrays to store points and colors
    #         num_points = width * height
    #         points = []
    #         colors = []
    #         time0 = time.time()
    #         for i in range(num_points):
    #             # Calculate point start position
    #             start_idx = i * point_step
                
    #             # Parse x, y, z coordinates
    #             x = struct.unpack_from('f', ros_pc2.data, start_idx + x_offset)[0]
    #             y = struct.unpack_from('f', ros_pc2.data, start_idx + y_offset)[0]
    #             z = struct.unpack_from('f', ros_pc2.data, start_idx + z_offset)[0]
                
    #             # Skip invalid points
    #             if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
    #                 continue
                
    #             points.append([x, y, z])
                
    #             # Parse color (if exists)
    #             if rgb_offset is not None:
    #                 rgb = struct.unpack_from('I', ros_pc2.data, start_idx + rgb_offset)[0]
    #                 r = (rgb >> 16) & 0xFF
    #                 g = (rgb >> 8) & 0xFF
    #                 b = rgb & 0xFF
    #                 colors.append([r/255.0, g/255.0, b/255.0])  # Color range in Open3D is 0-1
    #             else:
    #                 # Default to white
    #                 colors.append([1.0, 1.0, 1.0])
    #         time1 = time.time()
    #         # Create Open3D point cloud
    #         o3d_pc = o3d.geometry.PointCloud()
    #         if len(points) > 0:
    #             o3d_pc.points = o3d.utility.Vector3dVector(np.array(points))
    #             if len(colors) > 0:
    #                 o3d_pc.colors = o3d.utility.Vector3dVector(np.array(colors))
    #             time2 = time.time()
    #             print("[DEBUG]01:", (time1-time0)*1000, "ms")
    #             print("[DEBUG]12:", (time2-time1)*1000, "ms")
    #             # self.node.get_logger().info(f"Converted ROS2 PointCloud2 to Open3D point cloud with {len(points)} points")
    #             return o3d_pc
    #         else:
    #             self.node.get_logger().warning("Point cloud has no valid points")
    #             return None
    #     except Exception as e:
    #         self.node.get_logger().error(f"Error converting ROS2 PointCloud2 to Open3D: {str(e)}")
    #         return None

    def simple_ros_pointcloud2_to_o3d(self, ros_pc2):
        """
        Simplified version of converting ROS2 PointCloud2 message to Open3D point cloud
        Only extract coordinates, don't process colors
        """
        try:
            # Create an empty Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            
            # Get point cloud dimensions
            num_points = ros_pc2.width * ros_pc2.height
            
            # Create numpy array to store point coordinates
            points_list = []
            
            # Find field offsets
            x_offset = y_offset = z_offset = None
            for field in ros_pc2.fields:
                if field.name == 'x':
                    x_offset = field.offset
                elif field.name == 'y':
                    y_offset = field.offset
                elif field.name == 'z':
                    z_offset = field.offset
            
            if x_offset is None or y_offset is None or z_offset is None:
                self.node.get_logger().error("Point cloud missing coordinate fields")
                return None
            
            # Iterate through point cloud data
            time0 = time.time()
            for i in range(num_points):
                offset = i * ros_pc2.point_step
                
                # Extract coordinates
                try:
                    x = struct.unpack_from('f', ros_pc2.data, offset + x_offset)[0]
                    y = struct.unpack_from('f', ros_pc2.data, offset + y_offset)[0]
                    z = struct.unpack_from('f', ros_pc2.data, offset + z_offset)[0]
                    
                    # Only add valid points
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                        points_list.append([x, y, z])
                except Exception as e:
                    # Skip unparseable points
                    continue
            time1 = time.time()
            print("[DEBUG]process time: ", (time1-time0)*1000, "ms")
            # Add point coordinates to point cloud
            if len(points_list) > 0:
                pcd.points = o3d.utility.Vector3dVector(np.array(points_list))
                
                # Add uniform red color to point cloud
                colors = np.ones((len(points_list), 3)) * np.array([1, 0, 0])  # Red color
                pcd.colors = o3d.utility.Vector3dVector(colors)
                
                self.node.get_logger().info(f"Converted point cloud with {len(points_list)} points (simplified method)")
                return pcd
            else:
                self.node.get_logger().warning("No valid points found in point cloud")
                return None
        except Exception as e:
            self.node.get_logger().error(f"Error in simplified conversion: {str(e)}")
            return None
    
    def save_callback(self, msg: Bool):
        self.node.get_logger().info('save callback')
        if self.data_buffer.item is None:
            self.data_buffer.item = {
                'id': self.data_buffer.current_id,
                'mean': 0,    # nparray, 4
                'cov': 0,      # nparray, 4x4
                'weights': 0, # scalar, 1
                'best_pose': self.best_pose_mapFrame,
                'success': True,    # doesn't matter, will overwrite.
            }
        else:
            self.data_buffer.item['success'] = msg.data
        self.save_prediction_data(self.data_buffer.item)
    
    def scene_callback(self, msg: PointCloud2):
        # Try to convert using standard method
        try:
            o3d_cloud = self.ros_pointcloud2_to_o3d(msg)
        except Exception as e:
            self.node.get_logger().warning(f"Error in standard conversion: {str(e)}, trying simplified method")
            o3d_cloud = self.simple_ros_pointcloud2_to_o3d(msg)
            
        if o3d_cloud and len(o3d_cloud.points) > 0:
            self.scene = o3d_cloud
        else:
            self.node.get_logger().warning("Failed to convert point cloud or point cloud is empty")

    def theta_valid_check(self, pose, y_center=0.0):
        x, y, theta, torso1 = pose
        target_x = 2.0
        target_y = y_center
        
        # Calculate robot's orientation vector (unit vector in direction of theta)
        robot_orientation_x = np.cos(theta)
        robot_orientation_y = np.sin(theta)
        
        # Calculate vector from robot to target
        to_target_x = target_x - x
        to_target_y = target_y - y
        
        # Normalize the to_target vector
        to_target_length = np.sqrt(to_target_x**2 + to_target_y**2)
        if to_target_length > 0:
            to_target_x /= to_target_length
            to_target_y /= to_target_length
        
        # Calculate dot product (gives cosine of angle between vectors)
        dot_product = robot_orientation_x * to_target_x + robot_orientation_y * to_target_y
        
        # Calculate angle in radians
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Check if angle is within ±30 degrees (±π/6 radians)
        print(f'angle: {angle}, to_target_length: {to_target_length}')
        if angle > np.deg2rad(30) or to_target_length > 1.0:
            return False
        
        return True

    def setup_policy(self):
        pass

    def reset_callback(self, msg: Empty):
        self.node.get_logger().info('reset callback.')
        self.best_pose = [100.0, 0.0, 0.0, 7.85394366e-01]  # trick, just make it disappear from scene

    def sample_pose_callback(self, msg: Empty):
        global xmin, xmax, ymin, ymax, z_constant, torso1_constant
        self.node.get_logger().info('sample pose callback')
        is_theta_valid = False
        while not is_theta_valid:
            pose = []
            # Use macro-defined workspace area range
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            theta = np.random.uniform(-HALF_THETA, HALF_THETA)
            torso1 = torso1_constant + np.random.uniform(-HALF_TORSO1, HALF_TORSO1)
            if torso1 < torso1_min:
                torso1 = torso1_min
            elif torso1 > torso1_max:
                torso1 = torso1_max
            pose = [x, y, theta, torso1]
            is_theta_valid = self.theta_valid_check(pose, y_center=(ymin+ymax)/2)
            print(f'pose: {pose}, is_theta_valid: {is_theta_valid}')
        self.best_pose = pose
        self.best_pose_mapFrame = self.best_pose    # already under FRAME_ID.
    
    def base_se2_to_map_se2(self):
        base_x, base_y, base_theta = self.data_buffer.current_se2
        
        # Calculate the transformed pose
        predicted_x = self.best_pose[0]
        predicted_y = self.best_pose[1]
        predicted_theta = self.best_pose[2]
        
        # Rotate the predicted position by the base orientation
        x_map = base_x + predicted_x * np.cos(base_theta) - predicted_y * np.sin(base_theta)
        y_map = base_y + predicted_x * np.sin(base_theta) + predicted_y * np.cos(base_theta)
        # Add the angles
        theta_map = base_theta + predicted_theta
        
        # Normalize theta to [-pi, pi]
        theta_map = (theta_map + np.pi) % (2 * np.pi) - np.pi
        
        self.best_pose_mapFrame[0] = x_map
        self.best_pose_mapFrame[1] = y_map
        self.best_pose_mapFrame[2] = theta_map
        
    def sir_execute_callback(self, msg: Empty):
        control = RobotControl()
        control.command_type = "torso"
        torso_qpos = [0.00000000e+00,  7.85394366e-01, -1.57079253e+00,  7.85398163e-01,  0.00000000e+00,  0.00000000e+00]
        torso_qpos[1] = self.best_pose[3]
        torso_qpos[2] = -2*self.best_pose[3]
        torso_qpos[3] = self.best_pose[3]
        control.torso_qpos = [float(q) for q in torso_qpos]
        self.torso_control_pub.publish(control)
        self.node.get_logger().info(f'control torso: {self.best_pose[3]}')

    def sir_inference_callback(self, msg: Empty):
        self.node.get_logger().info('SIR inference begin.')
        self.node.get_logger().info(f'Current experiment ID: {self.data_buffer.current_id}')
        
        self.node.get_logger().info('convert obs to o3d')
        self.obs = self.ros_pointcloud2_to_o3d(self.data_buffer.pcl_rendered)

        if self.obs is not None and len(self.obs.points) > 0:
            self.node.get_logger().info(f'Using Open3D point cloud with {len(self.obs.points)} points')
            
            # target_helper = TargetHelper(
            #     pcd=self.obs,
            #     origin_se2=None,
            #     x_half_range=None,
            #     y_half_range=None,
            #     theta_half_range_deg=None
            # )
            self.node.get_logger().info(f'predict_SIR_target_point begin')
            
            # Use current experiment ID from client_helper_stitch if available
            save_id = self.data_buffer.current_id
            
            # self.best_pose, mean, cov, weights, _, _ = predict_SIR_target_point(
            #     SIR_predictor=self.predictor,
            #     SIR_config={'dataset':{'pointnum': 8192}},
            #     pc_numpy=np.concatenate([self.obs.points, self.obs.colors], axis=1),
            #     target_helper=target_helper,
            #     SIR_sample_num=300,
            #     task_name=None,
            #     save_dir=f'./n2m_inference/{self.task}',
            #     id=save_id
            # )
            self.best_pose, mean, cov, weights = predict_SIR_mean(
                SIR_predictor=self.predictor,
                SIR_config={'dataset':{'pointnum': 8192}},
                pc_numpy=np.concatenate([self.obs.points, self.obs.colors], axis=1),
                target_helper=None,
                SIR_sample_num=300,
                task_name=None,
                save_dir=f'./n2m_inference/{self.task}',
                id=save_id
            )
            
            self.base_se2_to_map_se2()  # change best_pose to FRAME_ID
            
            self.node.get_logger().info(f'Prediction success, best_pose: {self.best_pose}, saved with ID: {save_id}')
            print(save_id)
            
            self.data_buffer.item = {
                'id': save_id,
                'mean': mean[0],    # nparray, 4
                'cov': cov[0],      # nparray, 4x4
                'weights': weights[0], # scalar, 1
                'best_pose': self.best_pose_mapFrame,
                'success': True,    # doesn't matter, will overwrite.
            }
        else:
            self.node.get_logger().warning('No valid point cloud available for inference')
        
        self.node.get_logger().info('SIR inference end.')

    def tf_broadcaster_callback(self):
        try:
            # Check if we have received valid SE2 data
            if self.data_buffer.has_valid_se2:
                # Get current SE2 from buffer
                x, y, theta, _ = self.best_pose_mapFrame

                # Publish transformed pose
                publish_odom_to_base_tf(self.tf_broadcaster, float(x), float(y), float(theta), 
                                       self.clock, frameID=FRAME_ID)

                # Log occasionally to avoid flooding
                if not hasattr(self, 'last_tf_log_time') or (time.time() - self.last_tf_log_time) > 999.0:
                    self.node.get_logger().info(f'Using SE2-transformed pose: ({x:.3f}, {y:.3f}, {theta:.3f})')
                    self.last_tf_log_time = time.time()
            else:
                # If no valid SE2 data, use original values
                publish_odom_to_base_tf(self.tf_broadcaster, float(self.best_pose[0]), float(self.best_pose[1]), 
                                      float(self.best_pose[2]), self.clock, frameID="base")

            # Always publish joint transforms. best_pose and best_pose_mapframe share same torso height.
            publish_joint_transforms(self.tf_broadcaster, float(self.best_pose[3]), self.clock)
            
            # Merge two lists
            se2_source_and_target = list(self.data_buffer.current_se2) + list(self.best_pose_mapFrame[0:3])

            # Construct message
            msg = Float32MultiArray()
            msg.data = se2_source_and_target

            # Publish
            self.se2_nav_pub.publish(msg)
        except Exception as e:
            self.node.get_logger().error(f'Error in tf_broadcaster_callback: {str(e)}')
            # Fallback to original behavior
            publish_odom_to_base_tf(self.tf_broadcaster, float(self.best_pose[0]), float(self.best_pose[1]), 
                                  float(self.best_pose[2]), self.clock, frameID="base")
            publish_joint_transforms(self.tf_broadcaster, float(self.best_pose[3]), self.clock)
        
    def pcl_rendered_callback(self, msg: PointCloud2):
        # Store raw message
        self.data_buffer.pcl_rendered = msg
        
    def publish_workspace_boundary(self):
        """
        Create and publish rectangle marker representing workspace area
        """
        global xmin, xmax, ymin, ymax, z_constant
        marker = Marker()
        marker.header.frame_id = FRAME_ID  # FRAME_ID
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "workspace_boundary"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Set marker dimensions
        marker.scale.x = 0.05  # Line width
        
        # Set marker color (yellow)
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Rectangle vertices (clockwise direction, Z=0)
        marker.points = []
        
        # Add four corner points and return to start to form closed loop
        p1 = self.create_point(xmin, ymin, z_constant)
        p2 = self.create_point(xmax, ymin, z_constant)
        p3 = self.create_point(xmax, ymax, z_constant)
        p4 = self.create_point(xmin, ymax, z_constant)

        marker.points.append(p1)
        marker.points.append(p2)
        marker.points.append(p3)
        marker.points.append(p4)
        marker.points.append(p1)  # Return to start to form closed loop

        # Set marker lifetime
        marker.lifetime.sec = 1  # Automatically disappear after 1 second, timer updates once per second

        # Publish marker
        self.marker_pub.publish(marker)

        # Optional: add marker center point and label
        self.publish_workspace_center()

    def create_point(self, x, y, z):
        """Create a geometry message point"""
        p = Point()
        p.x = x
        p.y = y
        p.z = z
        return p

    def publish_workspace_center(self):
        global xmin, xmax, ymin, ymax, z_constant
        """
        Publish workspace center point marker and text label
        """
        # Calculate center point
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0

        # Create center point marker
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "workspace_center"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = center_x
        marker.pose.position.y = center_y
        marker.pose.position.z = z_constant
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.lifetime.sec = 1
        
        self.marker_pub.publish(marker)
        
        # Create text label
        text_marker = Marker()
        text_marker.header.frame_id = FRAME_ID
        text_marker.header.stamp = self.node.get_clock().now().to_msg()
        text_marker.ns = "workspace_label"
        text_marker.id = 2
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        text_marker.pose.position.x = center_x
        text_marker.pose.position.y = center_y
        text_marker.pose.position.z = z_constant + 0.2  # Text displayed above
        text_marker.pose.orientation.w = 1.0
        
        text_marker.scale.z = 0.1  # Text size
        
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        
        text_marker.text = f"Workspace: X[{xmin:.1f}-{xmax:.1f}], Y[{ymin:.1f}-{ymax:.1f}]"
        
        text_marker.lifetime.sec = 1
        
        self.marker_pub.publish(text_marker)

    def row_callback(self, msg: Empty):
        global current_row, current_col, xmin, xmax, ymin, ymax, z_constant, target_torso1, target_x, target_y, torso1_constant
        current_row = (current_row + 1) % 3  # Cycle through 0, 1, 2xmin = target_x[current_col] - HALF_X
        xmax = target_x[current_col] + HALF_X
        ymin = target_y[current_col] - HALF_Y
        ymax = target_y[current_col] + HALF_Y
        torso1_constant = target_torso1[current_row]
        self.node.get_logger().info(f'Changed row to: {current_row}, currently: row:{current_row} col:{current_col}')
        
    def col_callback(self, msg: Empty):
        global current_row, current_col, xmin, xmax, ymin, ymax, z_constant, target_torso1, target_x, target_y, torso1_constant
        current_col = (current_col + 1) % 4  # Cycle through 0, 1, 2, 3
        xmin = target_x[current_col] - HALF_X
        xmax = target_x[current_col] + HALF_X
        ymin = target_y[current_col] - HALF_Y
        ymax = target_y[current_col] + HALF_Y
        torso1_constant = target_torso1[current_row]
        self.node.get_logger().info(f'Changed col to: {current_col}, currently: row:{current_row} col:{current_col}')

    def id_callback(self, msg: Int32):
        """
        Callback for receiving the experiment ID from client_helper_stitch
        """
        self.data_buffer.current_id = msg.data
        self.node.get_logger().debug(f'Received experiment ID: {self.data_buffer.current_id}')

    def load_prediction_data(self):
        """
        Load existing prediction data from YAML file if it exists
        """
        try:
            if os.path.exists(self.prediction_yaml_path):
                with open(self.prediction_yaml_path, 'r') as f:
                    self.prediction_data = yaml.safe_load(f) or {}
                self.node.get_logger().info(f'Loaded {len(self.prediction_data)} predictions from {self.prediction_yaml_path}')
            else:
                self.prediction_data = {}
                self.node.get_logger().info(f'No existing prediction file found, will create new one when data is available')
        except Exception as e:
            self.node.get_logger().error(f'Error loading prediction data: {str(e)}')
            self.prediction_data = {}
    
    def save_prediction_data(self, item):
        """
        Save prediction data to YAML file
        """
        try:
            # Convert numpy arrays to lists for YAML serialization
            serializable_item = {}
            for key, value in item.items():
                if isinstance(value, np.ndarray):
                    serializable_item[key] = value.tolist()
                elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                    # Handle lists of numpy arrays
                    serializable_item[key] = [arr.tolist() for arr in value]
                elif isinstance(value, list) and value and isinstance(value[0], (float, int)) and hasattr(value, '__iter__'):
                    # Handle lists of numbers
                    serializable_item[key] = list(value)
                else:
                    serializable_item[key] = value
            
            # Update or add item by ID
            item_id = str(serializable_item['id'])  # Convert ID to string for YAML key
            self.prediction_data[item_id] = serializable_item
            
            # Save to file
            with open(self.prediction_yaml_path, 'w') as f:
                yaml.dump(self.prediction_data, f, default_flow_style=False)
            
            self.node.get_logger().info(f'Saved prediction data with ID {item_id} to {self.prediction_yaml_path}')
        except Exception as e:
            self.node.get_logger().error(f'Error saving prediction data: {str(e)}')
            # Save a backup in case of corruption
            try:
                backup_path = f"{self.prediction_yaml_path}.bak"
                with open(backup_path, 'w') as f:
                    yaml.dump(self.prediction_data, f, default_flow_style=False)
                self.node.get_logger().info(f'Saved backup to {backup_path}')
            except Exception as backup_error:
                self.node.get_logger().error(f'Error saving backup: {str(backup_error)}')

def main(task, args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('SIR_inference')
    SIR(node, task)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='laptop')
    args = parser.parse_args()
    main(task=args.task)