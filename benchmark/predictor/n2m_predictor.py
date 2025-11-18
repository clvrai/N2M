"""N2M predictor - predicts poses from point cloud using GMM.

Extracts core N2M prediction logic.
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional, Any
import torch
import open3d as o3d

from benchmark.predictor.base import BasePredictor
from benchmark.utils.observation_utils import observation_to_pointcloud, get_camera_params, fix_point_cloud_size
from benchmark.utils.collision_utils import CollisionChecker
from benchmark.utils.sampling_utils import sample_from_gmm, select_best_sample


class N2MPredictor(BasePredictor):
    """N2M predictor using point cloud → GMM → sampled pose.
    
    One-shot predictor that returns done=True on first call.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize N2M predictor.
        
        Args:
            config: Configuration dictionary with:
                - checkpoint_path: Path to N2M model checkpoint
                - camera_names: List of camera names for observation
                - num_points: Number of points in point cloud (default 1024)
                - num_samples: Number of samples from GMM (default 100)
                - use_collision_check: Whether to filter colliding samples
        """
        super().__init__(config)
        
        self.camera_names = config.get('camera_names', ['robot0_agentview_left'])
        self.num_points = config.get('num_points', 1024)
        self.num_samples = config.get('num_samples', 100)
        self.use_collision_check = config.get('use_collision_check', True)
        
        # N2M model (will be loaded in load_checkpoint)
        self.n2m_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Collision checker (built on first predict call)
        self.collision_checker = None
        
    def predict(
        self, 
        observation: Dict[str, np.ndarray],
        current_pose: np.ndarray,
        env_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Predict target pose using N2M.
        
        Args:
            observation: Observation dict with RGB and depth
            current_pose: Current robot base SE2 pose [x, y, theta]
            env_info: Environment info (contains target object position)
            
        Returns:
            predicted_pose: Sampled SE2 pose from GMM
            done: Always True (one-shot predictor)
            info: Dict with GMM distribution, samples, etc.
        """
        start_time = time.time()
        
        # Get camera parameters (cache them)
        if not hasattr(self, '_camera_intrinsics'):
            self._camera_intrinsics = {}
            self._camera_extrinsics = {}
            # Note: env needs to be passed somehow - we'll get it from observation context
            # For now assume they're in config or we compute on-the-fly
        
        # 1. Generate point cloud from observation
        # For simplicity, assume we have a helper in observation that already processed it
        # In real usage, we'd call observation_to_pointcloud here
        
        # Placeholder: In actual implementation, you'd extract point cloud
        # For now, create a simple implementation
        point_cloud = self._observation_to_n2m_pointcloud(observation, current_pose, env_info)
        
        # 2. Run N2M inference: point cloud → GMM distribution
        gmm_means, gmm_covs, gmm_weights = self._run_n2m_inference(point_cloud)
        
        # 3. Build collision checker if needed
        if self.use_collision_check and self.collision_checker is None:
            # Build from current point cloud
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            self.collision_checker = CollisionChecker(pcd_o3d)
        
        # 4. Sample from GMM and filter by collision
        samples, scores = sample_from_gmm(
            means=gmm_means,
            covariances=gmm_covs,
            weights=gmm_weights,
            num_samples=self.num_samples,
            collision_checker=self.collision_checker if self.use_collision_check else None
        )
        
        # 5. Select best sample
        if len(samples) == 0:
            # No valid samples, return current pose
            predicted_pose = current_pose.copy()
            is_valid = False
        else:
            best_idx = select_best_sample(samples, scores, selection_mode='max_score')
            predicted_pose = samples[best_idx]
            is_valid = True
        
        prediction_time = time.time() - start_time
        
        # 6. Build info dict
        info = {
            'distribution': {
                'means': gmm_means.tolist(),
                'covariances': gmm_covs.tolist(),
                'weights': gmm_weights.tolist()
            },
            'samples': samples.tolist() if len(samples) > 0 else [],
            'scores': scores.tolist() if len(scores) > 0 else [],
            'num_valid_samples': len(samples),
            'is_valid': is_valid,
            'prediction_time': prediction_time
        }
        
        return predicted_pose, True, info
    
    def _observation_to_n2m_pointcloud(
        self, 
        observation: Dict[str, np.ndarray],
        current_pose: np.ndarray,
        env_info: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Convert observation to N2M input format point cloud.
        
        Returns:
            point_cloud: (N, 6) point cloud [xyz, rgb] in robot frame
        """
        # Use utility function to generate point cloud from RGB-D
        from benchmark.env.env_utils import get_env_observation_with_depth
        
        # Get camera parameters (cached)
        if not hasattr(self, '_camera_params_cache'):
            from benchmark.utils.observation_utils import get_camera_params
            # Note: env is not directly accessible here
            # In practice, camera params should be passed in config or env_info
            # For now, use default RoboCasa camera params
            self._camera_params_cache = {}
        
        # Convert RGB-D to point cloud
        pcd = observation_to_pointcloud(
            observation=observation,
            camera_names=self.camera_names,
            camera_intrinsics=self._camera_intrinsics,
            camera_extrinsics=self._camera_extrinsics,
            num_points=self.num_points
        )
        
        # Convert Open3D point cloud to numpy array [xyz, rgb]
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        point_cloud = np.concatenate([points, colors], axis=1).astype(np.float32)
        
        # Ensure correct size
        point_cloud = fix_point_cloud_size(point_cloud, self.num_points)
        
        return point_cloud
    
    def _run_n2m_inference(self, point_cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run N2M model inference.
        
        Args:
            point_cloud: (N, 6) point cloud [xyz, rgb]
            
        Returns:
            means: (K, 3) GMM means [x, y, theta]
            covariances: (K, 3, 3) GMM covariances
            weights: (K,) GMM weights
        """
        if self.n2m_model is None:
            raise RuntimeError("N2M model not loaded. Call load_checkpoint() first.")
        
        # N2M model expects point cloud input
        # The model outputs GMM distribution parameters
        # Use the N2Mmodule's predict method
        with torch.no_grad():
            # N2Mmodule.predict() expects numpy array and returns SE2 pose + validity
            # But we need the GMM distribution, so we access internal methods
            output_dict = self.n2m_model.forward_inference(point_cloud)
        
        # Extract GMM parameters from output
        # Based on N2M model structure: outputs means, log_vars, weights
        means = output_dict['means'].cpu().numpy()  # (K, 3) for SE2
        log_vars = output_dict['log_vars'].cpu().numpy()  # (K, 3)
        weights = output_dict['weights'].cpu().numpy()  # (K,)
        
        # Convert log_vars to covariance matrices (diagonal)
        variances = np.exp(log_vars)
        covs = np.array([np.diag(var) for var in variances])
        
        return means, covs, weights
    
    def reset(self):
        """Reset predictor state."""
        # Collision checker can be reused across episodes
        pass
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load N2M model checkpoint.
        
        Args:
            checkpoint_path: Path to N2M checkpoint or config
        """
        # Import N2M module
        try:
            from n2m.module import N2Mmodule
        except ImportError:
            raise ImportError(
                "N2M module not found. Make sure predictor/N2M is installed: "
                "pip install -e predictor/N2M"
            )
        
        # Load model using N2Mmodule
        # N2Mmodule expects a config dict, not direct checkpoint path
        # Load config if checkpoint_path is a JSON config
        import json
        from pathlib import Path
        
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.suffix == '.json':
            # Load from config
            with open(checkpoint_path, 'r') as f:
                n2m_config = json.load(f)
            self.n2m_model = N2Mmodule(n2m_config)
        else:
            # Assume it's a direct checkpoint file
            # Create default config and load weights
            n2m_config = {
                'checkpoint_path': str(checkpoint_path),
                'device': str(self.device),
                'num_modes': 5
            }
            self.n2m_model = N2Mmodule(n2m_config)
        
        self.n2m_model.to(self.device)
        self.n2m_model.eval()
    
    @property
    def name(self) -> str:
        """Return predictor name."""
        return "n2m"
