#!/usr/bin/env python3
"""
Visualize Rollout Dataset

This script visualizes rollout datasets by overlaying robot SE(2) poses as colored arrows
on a point cloud of the scene. Blue arrows indicate successful rollouts (is_success=1.0)
and red arrows indicate failed rollouts (is_success=0.0).

Usage examples:
    # Basic visualization
    python visualize_rollouts.py --pcd_file path/to/scene.pcd --meta_json path/to/meta.json --output visualization.pcd
    
    # With detailed statistics
    python visualize_rollouts.py --pcd_file path/to/scene.pcd --meta_json path/to/meta.json --output visualization.pcd --show_stats
    
    # Custom arrow sizing and height
    python visualize_rollouts.py --pcd_file path/to/scene.pcd --meta_json path/to/meta.json --output visualization.pcd --interval 0.2 --z_value 1.0

The output is a point cloud file that can be viewed in tools like Open3D, CloudCompare, or similar.
"""

import numpy as np
import open3d as o3d
import json
import argparse
import os
from pathlib import Path


def load_point_cloud(file_path, use_color=True):
    """Load point cloud from PCD file"""
    pcd = o3d.io.read_point_cloud(file_path)
    point_cloud = np.asarray(pcd.points)

    if use_color:
        colors = np.asarray(pcd.colors)
        if colors.size > 0:
            point_cloud = np.concatenate([point_cloud, colors], axis=1)
        else:
            # If no colors, create default white colors
            colors = np.ones((point_cloud.shape[0], 3)) * 0.7
            point_cloud = np.concatenate([point_cloud, colors], axis=1)
    return point_cloud


def load_meta_data(meta_json_path):
    """Load meta.json file and extract episode data"""
    with open(meta_json_path, 'r') as f:
        data = json.load(f)
    
    episodes = data.get('episodes', [])
    poses_and_success = []
    
    for episode in episodes:
        se2_pose = episode['pose']['se2']  # [x, y, theta]
        is_success = episode['is_success']  # 1.0 for success, 0.0 for failure
        poses_and_success.append((se2_pose, is_success))
    
    return poses_and_success


def create_arrow_for_pose(se2_pose, is_success, interval=0.1, z_value=0.8):
    """Create an arrow geometry for a SE(2) pose, mimicking save_gmm_visualization_se2"""
    x, y, theta = se2_pose
    
    # Create arrow geometry
    arrow_length = interval * 2  # Make arrows a bit larger for visibility
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=interval/12,
        cone_radius=interval/8,
        cylinder_height=arrow_length*0.8,
        cone_height=arrow_length*0.2
    )
    
    # Rotate arrow to correct orientation
    # First rotate to align with x-axis
    R_x = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    
    # Then rotate by theta
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    arrow.rotate(R_x, center=[0, 0, 0])  # rotate to x-axis
    arrow.rotate(R, center=[0, 0, 0])    # rotate to theta
    arrow.translate([x, y, z_value])
    
    # Color arrow based on success: blue for success (1.0), red for failure (0.0)
    if is_success == 1.0:
        arrow.paint_uniform_color([0, 0, 1])  # Blue for success
    else:
        arrow.paint_uniform_color([1, 0, 0])  # Red for failure
    
    return arrow


def show_dataset_statistics(poses_and_success):
    """Show detailed statistics about the dataset"""
    success_poses = [pose for pose, success in poses_and_success if success == 1.0]
    failure_poses = [pose for pose, success in poses_and_success if success == 0.0]
    
    all_poses = [pose for pose, _ in poses_and_success]
    
    if len(all_poses) > 0:
        poses_array = np.array(all_poses)
        print("\n--- Dataset Statistics ---")
        print(f"Total episodes: {len(poses_and_success)}")
        print(f"Success episodes: {len(success_poses)}")
        print(f"Failure episodes: {len(failure_poses)}")
        success_rate = len(success_poses) / len(poses_and_success) * 100
        print(f"Success rate: {success_rate:.1f}%")
        
        print(f"\nPose distribution:")
        print(f"X range: [{poses_array[:, 0].min():.2f}, {poses_array[:, 0].max():.2f}]")
        print(f"Y range: [{poses_array[:, 1].min():.2f}, {poses_array[:, 1].max():.2f}]")
        print(f"Theta range: [{poses_array[:, 2].min():.2f}, {poses_array[:, 2].max():.2f}] radians")
        print(f"X mean: {poses_array[:, 0].mean():.2f}, std: {poses_array[:, 0].std():.2f}")
        print(f"Y mean: {poses_array[:, 1].mean():.2f}, std: {poses_array[:, 1].std():.2f}")
        print(f"Theta mean: {poses_array[:, 2].mean():.2f}, std: {poses_array[:, 2].std():.2f}")


def visualize_rollout_dataset(pcd_file_path, meta_json_path, output_path, interval=0.1, z_value=0.8, show_stats=False):
    """
    Visualize rollout dataset with point cloud and SE(2) poses as colored arrows
    
    Args:
        pcd_file_path: Path to point cloud file (.pcd)
        meta_json_path: Path to meta.json file containing episode data
        output_path: Path to save the visualization
        interval: Interval for arrow sizing
        z_value: Z height for placing arrows
    """
    print(f"Loading point cloud from: {pcd_file_path}")
    point_cloud = load_point_cloud(pcd_file_path)
    
    print(f"Loading meta data from: {meta_json_path}")
    poses_and_success = load_meta_data(meta_json_path)
    
    print(f"Found {len(poses_and_success)} episodes")
    
    if show_stats:
        show_dataset_statistics(poses_and_success)
    
    # Create visualization geometries
    geometries = []
    
    # Add original point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, 0:3])
    if point_cloud.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
    geometries.append(pcd)
    
    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(coord_frame)
    
    # Create arrows for each pose
    success_count = 0
    failure_count = 0
    
    for se2_pose, is_success in poses_and_success:
        arrow = create_arrow_for_pose(se2_pose, is_success, interval, z_value)
        geometries.append(arrow)
        
        if is_success == 1.0:
            success_count += 1
        else:
            failure_count += 1
    
    print(f"Success episodes: {success_count} (blue arrows)")
    print(f"Failure episodes: {failure_count} (red arrows)")
    success_rate = success_count / (success_count + failure_count) * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    # Convert all geometries to point clouds and combine
    combined_points = []
    combined_colors = []
    
    for geom in geometries:
        if isinstance(geom, o3d.geometry.PointCloud):
            combined_points.append(np.asarray(geom.points))
            combined_colors.append(np.asarray(geom.colors))
        else:
            # Sample points from mesh
            pcd = geom.sample_points_uniformly(number_of_points=500)
            combined_points.append(np.asarray(pcd.points))
            combined_colors.append(np.asarray(pcd.colors))
    
    # Combine all points and colors
    all_points = np.vstack(combined_points)
    all_colors = np.vstack(combined_colors)
    all_colors = np.clip(all_colors, 0, 1)
    
    # Create final point cloud
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(all_points)
    final_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # Save to file
    print(f"Saving visualization to: {output_path}")
    o3d.io.write_point_cloud(output_path, final_pcd)
    print("Visualization saved successfully!")


def main():
    parser = argparse.ArgumentParser(description='Visualize rollout dataset with point cloud and SE(2) poses')
    parser.add_argument('--pcd_file', type=str, required=True,
                        help='Path to point cloud file (.pcd)')
    parser.add_argument('--meta_json', type=str, required=True,
                        help='Path to meta.json file containing episode data')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save the visualization (.pcd)')
    parser.add_argument('--interval', type=float, default=0.1,
                        help='Interval for arrow sizing (default: 0.1)')
    parser.add_argument('--z_value', type=float, default=0.8,
                        help='Z height for placing arrows (default: 0.8)')
    parser.add_argument('--show_stats', action='store_true',
                        help='Show additional statistics about the dataset')
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.pcd_file):
        print(f"Error: Point cloud file not found: {args.pcd_file}")
        return
    
    if not os.path.exists(args.meta_json):
        print(f"Error: Meta JSON file not found: {args.meta_json}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run visualization
    visualize_rollout_dataset(
        pcd_file_path=args.pcd_file,
        meta_json_path=args.meta_json,
        output_path=args.output,
        interval=args.interval,
        z_value=args.z_value,
        show_stats=args.show_stats
    )


if __name__ == "__main__":
    main()
