"""Data collection utilities for predictors."""

import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import open3d as o3d

from benchmark.utils.observation_utils import observation_to_pointcloud, get_camera_params
from benchmark.utils.transform_utils import obs_to_SE2
from benchmark.env.env_utils import get_target_object_info, get_env_observation_with_depth

class N2MDataCollector:
    """Collect training data for N2M predictor.
    
    Saves data in format required by N2M module:
    - meta.json: metadata for each trajectory
    - pcl/: point cloud files
    """
    
    def __init__(self, output_dir: str):
        """Initialize data collector.
        
        Args:
            output_dir: Output directory for collected data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.pcl_dir = self.output_dir / "pcl"
        self.pcl_dir.mkdir(exist_ok=True)
        
        # N2M format: separate global meta and episodes list
        self.global_meta = None  # Will be set on first episode
        self.episodes = []
        
        # Try to load existing metadata for incremental collection
        self._load_existing_metadata()
    
    def get_num_collected(self) -> int:
        """Get number of episodes already collected.
        
        Returns:
            Number of episodes in current collection
        """
        return len(self.episodes)
        
    def _load_existing_metadata(self):
        """Load existing metadata if available (for incremental collection)."""
        meta_path = self.output_dir / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    data = json.load(f)
                    self.global_meta = data.get('meta')
                    self.episodes = data.get('episodes', [])
                print(f"Loaded existing metadata: {len(self.episodes)} episodes")
            except Exception as e:
                print(f"Warning: Failed to load existing metadata: {e}")
                self.global_meta = None
                self.episodes = []
    
    def save_episode_pointcloud(
        self,
        pcd,
        episode_id: int,
        target_pose: np.ndarray,
        depth_cameras: List[str],
        env,
        algo_name: str = "bc"
    ):
        """Save point cloud and metadata for one successful episode.
        
        This method should be called AFTER rollout succeeds.
        The point cloud should have been captured BEFORE rollout (when robot was moved away).
        
        Args:
            pcd: Open3D point cloud (already captured before rollout)
            episode_id: Episode ID
            target_pose: Target SE2 pose that robot reached
            depth_cameras: List of depth camera names used
            env: RoboCasa environment
            algo_name: Algorithm name (e.g., 'act', 'bc') to determine unwrapping layers
        """
        # Get unwrapped environment
        is_act_policy = algo_name == "act"
        if is_act_policy:
            unwrapped_env = env.env
        else:
            unwrapped_env = env.env.env
        
        # Save point cloud with N2M naming: {id}.pcd
        pcl_filename = f"{episode_id}.pcd"
        pcl_path = self.pcl_dir / pcl_filename
        o3d.io.write_point_cloud(str(pcl_path), pcd)
        
        # Get camera parameters from first depth camera (for global meta)
        # Set global meta on first episode only (shared across all episodes)
        if self.global_meta is None:
            first_cam = depth_cameras[0]
            intrinsic, extrinsic = get_camera_params(env, first_cam)
            
            # Get unwrapped environment (following reference implementation)
            temp_unwrapped = env
            while hasattr(temp_unwrapped, 'env'):
                temp_unwrapped = temp_unwrapped.env
            
            # Get camera dimensions from env
            cam_idx = temp_unwrapped.camera_names.index(first_cam)
            cam_height = temp_unwrapped.camera_heights[cam_idx]
            cam_width = temp_unwrapped.camera_widths[cam_idx]
            
            # Set global meta (N2M format)
            self.global_meta = {
                'T_base_to_cam': extrinsic.tolist(),
                'camera_intrinsic': [
                    intrinsic[0, 0],  # fx
                    intrinsic[1, 1],  # fy
                    intrinsic[0, 2],  # cx
                    intrinsic[1, 2],  # cy
                    cam_width,        # width
                    cam_height,       # height
                ]
            }
        
        # Convert SE2 pose [x, y, theta] to N2M format [x, y, z, theta]
        # N2M format expects 4D pose: [x, y, z, theta]
        # target_pose from SE2 is [x, y, theta], we add z coordinate
        robot_x, robot_y, robot_theta = target_pose
        robot_z = 0.0  # Base z position (robot base is at ground level)
        
        # Create episode entry (N2M standard format)
        # Format: {"id": int, "pose": [x, y, z, theta], "file_path": str}
        episode_entry = {
            'id': episode_id,
            'pose': [robot_x, robot_y, robot_z, robot_theta],  # N2M format: [x, y, z, theta]
            'file_path': f'pcl/{pcl_filename}'  # Relative path
        }
        
        self.episodes.append(episode_entry)
        
    def save_metadata(self):
        """Save metadata to meta.json file in N2M standard format.
        
        This method should be called after each successful episode for incremental saving.
        
        N2M standard format:
        {
            "meta": {
                "T_base_to_cam": [[4x4 matrix]],
                "camera_intrinsic": [fx, fy, cx, cy, width, height]
            },
            "episodes": [
                {
                    "id": 0,
                    "pose": [x, y, z, theta],  # Robot pose (4D)
                    "file_path": "pcl/0.pcd"
                },
                ...
            ]
        }
        """
        # Create N2M format
        n2m_data = {
            'meta': self.global_meta,
            'episodes': self.episodes
        }
        
        meta_path = self.output_dir / "meta.json"
        with open(meta_path, 'w') as f:
            json.dump(n2m_data, f, indent=4)


class MobipiDataCollector:
    """Collect training data for Mobipi predictor (3DGS training).
    
    TODO: Implement image collection for 3DGS reconstruction.
    """
    
    def __init__(self, output_dir: str):
        """Initialize data collector.
        
        Args:
            output_dir: Output directory for collected data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        raise NotImplementedError(
            "Mobipi data collection not yet implemented. "
            "This requires collecting multi-view RGB images + camera poses "
            "for 3D Gaussian Splatting reconstruction."
        )
    
    def collect_episode(self, env, episode_id: int):
        """Collect one episode of Mobipi training data."""
        raise NotImplementedError()
