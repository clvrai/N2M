"""Observation processing utilities.

Migrated from A_ref/N2M-sim
"""

import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple, Optional


def observation_to_pointcloud(
    observation: Dict[str, np.ndarray],
    camera_names: List[str],
    camera_intrinsics: Dict[str, np.ndarray],
    camera_extrinsics: Dict[str, np.ndarray],
    num_points: int = 1024
) -> o3d.geometry.PointCloud:
    """Convert RGB-D observations to point cloud.
    
    Args:
        observation: Observation dictionary with RGB and depth images
        camera_names: List of camera names to use
        camera_intrinsics: Dict mapping camera name to intrinsic matrix (3x3)
        camera_extrinsics: Dict mapping camera name to extrinsic matrix (4x4)
        num_points: Target number of points (will downsample)
        
    Returns:
        Point cloud with color
    """
    all_points = []
    all_colors = []
    
    for cam_name in camera_names:
        # Get RGB and depth
        rgb_key = f"{cam_name}_rgb" if f"{cam_name}_rgb" in observation else f"{cam_name}_image"
        depth_key = f"{cam_name}_depth"
        
        if rgb_key not in observation or depth_key not in observation:
            continue
            
        rgb = observation[rgb_key]  # (H, W, 3), values in [0, 255] or [0, 1]
        depth = observation[depth_key]  # (H, W)
        
        # Normalize RGB if needed
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        # Get camera parameters
        K = camera_intrinsics[cam_name]
        T_world_cam = camera_extrinsics[cam_name]
        
        # Create point cloud from depth
        height, width = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Create meshgrid
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Compute 3D points in camera frame
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack into points
        points_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = rgb.reshape(-1, 3)
        
        # Filter out invalid points (zero depth)
        valid_mask = points_cam[:, 2] > 0
        points_cam = points_cam[valid_mask]
        colors = colors[valid_mask]
        
        # Transform to world frame
        points_cam_hom = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
        points_world_hom = (T_world_cam @ points_cam_hom.T).T
        points_world = points_world_hom[:, :3]
        
        all_points.append(points_world)
        all_colors.append(colors)
    
    # Combine all cameras
    if len(all_points) == 0:
        # Return empty point cloud
        return o3d.geometry.PointCloud()
    
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    # Downsample to target number of points
    if num_points is not None and len(pcd.points) > num_points:
        pcd = pcd.farthest_point_down_sample(num_points)
    
    return pcd


def get_camera_params(
    env, 
    camera_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Get camera intrinsic and extrinsic parameters.
    
    Args:
        env: RoboCasa environment
        camera_name: Name of the camera
        
    Returns:
        intrinsic: 3x3 intrinsic matrix
        extrinsic: 4x4 extrinsic matrix (world to camera)
    """
    from robosuite.utils.camera_utils import (
        get_camera_intrinsic_matrix,
        get_camera_extrinsic_matrix
    )
    
    # Get unwrapped environment (following reference implementation)
    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env
    
    # Get camera height and width from env
    camera_heights = unwrapped_env.camera_heights
    camera_widths = unwrapped_env.camera_widths
    camera_names_list = unwrapped_env.camera_names
    
    # Find camera index
    cam_idx = camera_names_list.index(camera_name)
    camera_height = camera_heights[cam_idx]
    camera_width = camera_widths[cam_idx]
    
    # Get intrinsic matrix
    intrinsic = get_camera_intrinsic_matrix(
        unwrapped_env.sim,
        camera_name,
        camera_height,
        camera_width
    )
    
    # Get extrinsic matrix
    extrinsic = get_camera_extrinsic_matrix(
        unwrapped_env.sim,
        camera_name
    )
    
    return intrinsic, extrinsic


def fix_point_cloud_size(
    pcd: o3d.geometry.PointCloud, 
    target_size: int
) -> o3d.geometry.PointCloud:
    """Fix point cloud to target size by downsampling or upsampling.
    
    Args:
        pcd: Input point cloud
        target_size: Target number of points
        
    Returns:
        Point cloud with exactly target_size points
    """
    current_size = len(pcd.points)
    
    if current_size == target_size:
        return pcd
    elif current_size > target_size:
        # Downsample
        return pcd.farthest_point_down_sample(target_size)
    else:
        # Upsample by duplicating points
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Randomly duplicate points
        indices = np.random.choice(current_size, size=target_size - current_size, replace=True)
        extra_points = points[indices]
        extra_colors = colors[indices]
        
        # Combine
        new_points = np.vstack([points, extra_points])
        new_colors = np.vstack([colors, extra_colors])
        
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_points)
        new_pcd.colors = o3d.utility.Vector3dVector(new_colors)
        
        return new_pcd
