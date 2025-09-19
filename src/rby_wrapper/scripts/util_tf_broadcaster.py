from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import tf2_ros
import numpy as np

# robot_name = ""
robot_name = "robot2_"

# map_or_odom
# FRAME_ID = "map"  
FRAME_ID = "odom"  

def publish_odom_to_base_tf(tf_broadcaster: tf2_ros.TransformBroadcaster, x, y, theta, clock, frameID=FRAME_ID):
    """
    Publish TF transform from odom to base frame
    """
    # Create transform message
    t = TransformStamped()
    t.header.stamp = clock.now().to_msg()
    t.header.frame_id = frameID
    t.child_frame_id = f"{robot_name}base"
    
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
    tf_broadcaster.sendTransform(t)
    
def publish_joint_transforms(tf_broadcaster: tf2_ros.TransformBroadcaster, torso1, clock):
    """
    Publish TF transforms for torso and arm joints based on joint positions and URDF info
    """
    joint_state = np.array([-9.26684186e-01, -2.02658037e+00,  0.00000000e+00,  7.85394366e-01,
                            -1.57079253e+00,  7.85398163e-01,  0.00000000e+00,  0.00000000e+00,
                            4.36326407e-01, -8.72698377e-02,  0.00000000e+00, -2.09440017e+00,
                            3.83495197e-06,  1.22172366e+00,  0.00000000e+00,  4.36322610e-01,
                            8.72698377e-02, -3.79698215e-06, -2.09438877e+00,  0.00000000e+00,
                            1.22172366e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
    joint_state[3] = torso1
    joint_state[4] = -2*torso1
    joint_state[5] = torso1
    
    # Get joint indices for each group
    torso_idx= [2, 3, 4, 5, 6, 7]
    left_arm_idx= [15, 16, 17, 18, 19, 20, 21]
    right_arm_idx= [8, 9, 10, 11, 12, 13, 14]

    # Get current time for all transforms
    current_time = clock.now().to_msg()
    
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
            t.header.frame_id = f"{robot_name}base"
            t.child_frame_id = f"{robot_name}link_torso_{i}"
        else:
            t.header.frame_id = f"{robot_name}link_torso_{i-1}"
            t.child_frame_id = f"{robot_name}link_torso_{i}"
        
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
        tf_broadcaster.sendTransform(t)
    
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
            t.header.frame_id = f"{robot_name}link_torso_5"  # Arm connects to the last torso link
            t.child_frame_id = f"{robot_name}link_left_arm_{i}"
        else:
            t.header.frame_id = f"{robot_name}link_left_arm_{i-1}"
            t.child_frame_id = f"{robot_name}link_left_arm_{i}"
        
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
        tf_broadcaster.sendTransform(t)
    
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
            t.header.frame_id = f"{robot_name}link_torso_5"  # Arm connects to the last torso link
            t.child_frame_id = f"{robot_name}link_right_arm_{i}"
        else:
            t.header.frame_id = f"{robot_name}link_right_arm_{i-1}"
            t.child_frame_id = f"{robot_name}link_right_arm_{i}"
        
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
        tf_broadcaster.sendTransform(t)
