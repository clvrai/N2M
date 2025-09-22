import open3d as o3d
import numpy as np
import sys
import os
from pathlib import Path
import argparse
import json
import re

def create_coordinate_frame(pose, size=1.0, color=None):
    """Create a coordinate frame at the specified pose"""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(pose)
    
    # Convert to point cloud
    frame_points = frame.sample_points_uniformly(number_of_points=1000)
    points = np.asarray(frame_points.points)
    
    # Set colors if specified
    if color is not None:
        colors = np.tile(color, (len(points), 1))
    else:
        colors = np.asarray(frame_points.colors)
    
    frame_pcd = o3d.geometry.PointCloud()
    frame_pcd.points = o3d.utility.Vector3dVector(points)
    frame_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return frame_pcd

def create_se2_arrow(se2_pose, size=1.0, color=[0, 0, 1]):  # Default blue color
    """Create an arrow representing SE2 pose"""
    # Extract position and orientation
    x, y, theta = se2_pose
    z = 1.0  # Set z to 1 by default
    
    # Create arrow pointing in +x direction
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.05,
        cone_radius=0.1,
        cylinder_height=size,
        cone_height=0.2
    )
    
    # Rotate arrow to point in +x direction by default
    R_to_x = np.array([
        [0, 0, 1],
        [0, 1, 0], 
        [1, 0, 0]
    ])
    init_transform = np.eye(4)
    init_transform[0:3, 0:3] = R_to_x
    arrow.transform(init_transform)
    
    # Create final transformation matrix
    transform = np.eye(4)
    transform[0:3, 3] = [x, y, z]  # Set position
    
    # Create rotation matrix for yaw
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    transform[0:3, 0:3] = rotation
    
    # Apply final transform
    arrow.transform(transform)
    
    # Convert to point cloud
    arrow_points = arrow.sample_points_uniformly(number_of_points=1000)
    points = np.asarray(arrow_points.points)
    colors = np.tile(color, (len(points), 1))
    
    arrow_pcd = o3d.geometry.PointCloud()
    arrow_pcd.points = o3d.utility.Vector3dVector(points)
    arrow_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return arrow_pcd

def visualize_point_cloud_with_poses(pcd_path, base_pose, camera_pose, se2_pose, output_path):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        print(f"Failed to load point cloud file: {pcd_path}")
        return

    # Create coordinate frames
    origin_frame = create_coordinate_frame(np.eye(4), size=1.0)  # Origin frame (default colors)
    base_frame = create_coordinate_frame(base_pose, size=0.8, color=[1, 0, 0])  # Base frame (red)
    camera_frame = create_coordinate_frame(camera_pose, size=0.8, color=[0, 1, 0])  # Camera frame (green)
    se2_arrow = create_se2_arrow(se2_pose, size=1.0, color=[0, 0, 1])  # SE2 arrow (blue)

    # Combine all point clouds
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(np.vstack((
        np.asarray(pcd.points),
        np.asarray(origin_frame.points),
        np.asarray(base_frame.points),
        np.asarray(camera_frame.points),
        np.asarray(se2_arrow.points)
    )))
    combined_pcd.colors = o3d.utility.Vector3dVector(np.vstack((
        np.asarray(pcd.colors),
        np.asarray(origin_frame.colors),
        np.asarray(base_frame.colors),
        np.asarray(camera_frame.colors),
        np.asarray(se2_arrow.colors)
    )))

    # Save combined point cloud
    o3d.io.write_point_cloud(str(output_path), combined_pcd)
    print(f"Saved combined point cloud to: {output_path}")

def parse_pose_file(pose_file):
    """Parse pose file and return list of poses"""
    with open(pose_file, 'r') as f:
        poses = json.load(f)
    return poses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd_dir", type=str, required=True, help="Directory containing PCD files")
    parser.add_argument("--camera_pose_file", type=str, required=True, help="JSON file containing camera poses")
    parser.add_argument("--base_pose_file", type=str, required=True, help="JSON file containing base poses")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output PCDs")
    parser.add_argument("--meta_file", type=str, required=True, help="JSON file containing meta information")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pose files and meta file
    camera_poses = parse_pose_file(args.camera_pose_file)
    base_poses = parse_pose_file(args.base_pose_file)

    view_num = len(camera_poses[0])
    
    with open(args.meta_file, 'r') as f:
        meta = json.load(f)

    # Process each PCD file
    for pcd_file in os.listdir(args.pcd_dir):
        if not pcd_file.endswith('.pcd'):
            continue

        # Parse indices from filename (format: {i}_{j}.pcd)
        match = re.match(r'(\d+)_(\d+)\.pcd', pcd_file)
        if not match:
            print(f"Skipping file with invalid format: {pcd_file}")
            continue

        i, j = map(int, match.groups())
        
        # Get corresponding poses
        try:
            camera_pose = np.array(camera_poses[i][j])
            base_pose = np.array(base_poses[i][j])
            se2_pose = np.array(meta["episodes"][i * view_num + j]["pose"]["se2"])  # Get SE2 pose from meta file
        except (IndexError, KeyError):
            print(f"Skipping file {pcd_file}: poses not found for indices {i}, {j}")
            continue

        # Process the point cloud
        pcd_path = os.path.join(args.pcd_dir, pcd_file)
        output_path = os.path.join(args.output_dir, f"{pcd_file[:-4]}_with_poses.pcd")
        
        visualize_point_cloud_with_poses(pcd_path, base_pose, camera_pose, se2_pose, output_path)

if __name__ == "__main__":
    main()