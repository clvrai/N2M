import yaml
import numpy as np
import open3d as o3d
import os
import math
import argparse

def load_yaml_data(file_path):
    """Load YAML data from file"""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def load_point_cloud(pcl_path):
    """Load point cloud from PCD file"""
    if not os.path.exists(pcl_path):
        print(f"Point cloud file not found: {pcl_path}")
        return None
    
    try:
        pcd = o3d.io.read_point_cloud(pcl_path)
        return pcd
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return None

def create_camera_pyramid(size=0.1):
    """Create a rectangular pyramid to represent a camera/robot"""
    # Define vertices for the pyramid
    vertices = np.array([
        [0, 0, 0],  # apex
        [-size/2, -size/2, size],  # base vertices
        [size/2, -size/2, size],
        [size/2, size/2, size],
        [-size/2, size/2, size]
    ])
    
    # Define triangles
    triangles = np.array([
        [0, 1, 2],  # front face
        [0, 2, 3],  # right face
        [0, 3, 4],  # back face
        [0, 4, 1],  # left face
        [1, 2, 3],  # base
        [1, 3, 4]
    ])
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    
    return mesh

def euler_to_rotation_matrix(theta):
    """Convert euler angle (theta) to rotation matrix for 2D rotation around Z axis"""
    R = np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
    ])
    return R

def create_robot_visualization(x, y, z, theta, is_success, size=0.2):
    """Create a colored pyramid representing the robot/camera at the specified pose"""
    # Create base pyramid
    pyramid = create_camera_pyramid(size)
    
    # Set color based on success/failure
    if is_success:
        color = [0, 1, 0]  # Green for success
    else:
        color = [1, 0, 0]  # Red for failure
    
    pyramid.paint_uniform_color(color)
    
    # Create transformation matrix
    R = euler_to_rotation_matrix(theta)
    t = np.array([x, y, z])
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    # Transform pyramid
    pyramid.transform(T)
    
    return pyramid

def create_head_camera_visualization(transform_matrix, size=0.15):
    """Create a visualization for the head camera using the transformation matrix"""
    # Create a pyramid for the camera
    pyramid = create_camera_pyramid(size)
    
    # Set color for head camera (blue)
    color = [0, 0, 1]  # Blue
    pyramid.paint_uniform_color(color)
    
    # Transform pyramid using the provided transformation matrix
    pyramid.transform(transform_matrix)
    
    return pyramid

def create_camera_axes(transform_matrix, size=0.2):
    """Create coordinate frame to visualize the camera axes"""
    # Create a coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
    # Transform the coordinate frame using the camera transformation matrix
    coord_frame.transform(transform_matrix)
    
    return coord_frame

def extract_transform_matrix(transform_data):
    """Extract 4x4 transformation matrix from transform data in YAML"""
    matrix = np.eye(4)
    
    matrix[0, 0] = transform_data['row0'][0]
    matrix[0, 1] = transform_data['row0'][1]
    matrix[0, 2] = transform_data['row0'][2]
    matrix[0, 3] = transform_data['row0'][3]
    
    matrix[1, 0] = transform_data['row1'][0]
    matrix[1, 1] = transform_data['row1'][1]
    matrix[1, 2] = transform_data['row1'][2]
    matrix[1, 3] = transform_data['row1'][3]
    
    matrix[2, 0] = transform_data['row2'][0]
    matrix[2, 1] = transform_data['row2'][1]
    matrix[2, 2] = transform_data['row2'][2]
    matrix[2, 3] = transform_data['row2'][3]
    
    matrix[3, 0] = transform_data['row3'][0]
    matrix[3, 1] = transform_data['row3'][1]
    matrix[3, 2] = transform_data['row3'][2]
    matrix[3, 3] = transform_data['row3'][3]
    
    return matrix

def create_arrow(start, direction, length=0.3, color=[0, 0, 1]):
    """Create an arrow to show orientation"""
    # Create a cylinder for the arrow body
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=length*0.8)
    
    # Create a cone for the arrow head
    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.02, height=length*0.2)
    
    # Position the cone at the end of the cylinder
    cone_transform = np.eye(4)
    cone_transform[2, 3] = length * 0.8
    cone.transform(cone_transform)
    
    # Combine cylinder and cone
    arrow = cylinder + cone
    
    # Rotate the arrow to align with the direction
    direction_norm = direction / np.linalg.norm(direction)
    
    # Create a rotation matrix to align with the direction
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction_norm)
    
    if np.linalg.norm(rotation_axis) < 1e-6:
        # If direction is parallel to z-axis
        if direction_norm[2] > 0:
            # Same direction as z-axis, no rotation needed
            R = np.eye(3)
        else:
            # Opposite direction, rotate 180 degrees around x-axis
            R = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
    else:
        # Normalize rotation axis
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Calculate rotation angle
        angle = np.arccos(np.dot(z_axis, direction_norm))
        
        # Rodrigues rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    # Create transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = start
    
    # Apply transformation
    arrow.transform(T)
    arrow.paint_uniform_color(color)
    
    return arrow

def visualize_data(yaml_data, pcl_dir, index=None):
    """Visualize data using Open3D"""
    geometries = []
    
    # Add coordinate frame for reference
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(world_frame)
    
    # Process entries
    indices = [index] if index is not None else yaml_data.keys()
    
    for idx in indices:
        if idx == '':
            continue
            
        if idx not in yaml_data:
            print(f"Index {idx} not found in YAML data")
            continue
            
        entry = yaml_data[idx]
        
        # Extract data for base position
        x = entry.get('x', 0)
        y = entry.get('y', 0)
        theta = entry.get('theta', 0)
        torso_1 = entry.get('torso_1', 0)
        success = entry.get('success', False)
        
        # Calculate z from torso angle
        z = 2 * math.tan(torso_1)
        
        # Load point cloud
        pcl_path = os.path.join(pcl_dir, f"{idx}.pcd")
        pcd = load_point_cloud(pcl_path)
        
        if pcd is not None:
            # Downsample the point cloud for better performance
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            geometries.append(pcd)
        
        # Create robot base visualization
        robot_vis = create_robot_visualization(x, y, z, theta, success)
        geometries.append(robot_vis)
        
        # Create arrow to show base orientation
        direction = np.array([math.cos(theta), math.sin(theta), 0])
        arrow = create_arrow([x, y, z], direction)
        geometries.append(arrow)
        
        # Visualize head camera transform if available
        if 'head_cam_transform' in entry:
            # Extract head camera transformation matrix
            head_cam_transform = extract_transform_matrix(entry['head_cam_transform'])
            
            # Create head camera visualization
            head_cam_vis = create_head_camera_visualization(head_cam_transform)
            geometries.append(head_cam_vis)
            
            # Add coordinate frame to show camera axes
            camera_axes = create_camera_axes(head_cam_transform)
            geometries.append(camera_axes)
            
            # No longer need a separate arrow since the coordinate frame shows all axes
        
        # Add text label for the index
        print(f"Index: {idx}, Position: ({x:.2f}, {y:.2f}, {z:.2f}), "
              f"Theta: {theta:.2f}, Success: {success}")
        if 'head_cam_transform' in entry:
            head_pos = entry['head_cam_transform']['row0'][3], entry['head_cam_transform']['row1'][3], entry['head_cam_transform']['row2'][3]
            print(f"Head Camera Position: ({head_pos[0]:.2f}, {head_pos[1]:.2f}, {head_pos[2]:.2f})")
            
            # Also print the camera orientation (rotation matrix)
            print("Camera Orientation (rows of rotation matrix):")
            print(f"  X-axis: [{head_cam_transform[0,0]:.2f}, {head_cam_transform[0,1]:.2f}, {head_cam_transform[0,2]:.2f}]")
            print(f"  Y-axis: [{head_cam_transform[1,0]:.2f}, {head_cam_transform[1,1]:.2f}, {head_cam_transform[1,2]:.2f}]")
            print(f"  Z-axis: [{head_cam_transform[2,0]:.2f}, {head_cam_transform[2,1]:.2f}, {head_cam_transform[2,2]:.2f}]")
    
    # Visualize
    o3d.visualization.draw_geometries(geometries)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize dataset with point cloud and robot poses')
    parser.add_argument('--yaml', type=str, default="dataset_rollout/lamp/meta_data.yaml", 
                        help='Path to YAML metadata file')
    parser.add_argument('--pcl_dir', type=str, default="dataset_rollout/lamp/pcl", 
                        help='Directory containing point cloud files')
    parser.add_argument('--index', type=int, default=None, 
                        help='Specific index to visualize (optional)')
    args = parser.parse_args()
    
    # Load YAML data
    yaml_data = load_yaml_data(args.yaml)
    
    # Visualize data
    visualize_data(yaml_data, args.pcl_dir, args.index)
    
if __name__ == "__main__":
    main()