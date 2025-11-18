"""Mobipi predictor - uses 3D Gaussian Splatting and Bayesian Optimization.

Reference: N2M-benchmark/predictor/mobipi/mobipi
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional, Any
import torch

from benchmark.predictor.base import BasePredictor


class MobipiPredictor(BasePredictor):
    """Mobipi predictor using 3DGS scene model and optimization.
    
    One-shot predictor that returns done=True on first call.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Mobipi predictor.
        
        Args:
            config: Configuration dictionary with:
                - scene_model_path: Path to pre-built 3DGS scene model
                - policy_checkpoint_path: Path to manipulation policy (for features)
                - encoder_type: Feature encoder ('dino_dense_descriptor', 'policy', 'dino')
                - num_candidates: Number of candidate poses to optimize (default 32)
                - optimization_steps: Number of optimization iterations (default 50)
                - rebuild_scene_model: Whether to rebuild 3DGS on-site (TODO, default False)
        """
        super().__init__(config)
        
        self.scene_model_path = config['scene_model_path']
        self.policy_checkpoint_path = config.get('policy_checkpoint_path', None)
        self.encoder_type = config.get('encoder_type', 'dino_dense_descriptor')
        self.num_candidates = config.get('num_candidates', 32)
        self.optimization_steps = config.get('optimization_steps', 50)
        self.rebuild_scene_model = config.get('rebuild_scene_model', False)
        
        # Mobipi components (loaded in load_checkpoint)
        self.scene_model = None
        self.feature_encoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.rebuild_scene_model:
            raise NotImplementedError(
                "On-site 3DGS reconstruction not yet implemented. "
                "Please use pre-built scene models for now."
            )
    
    def predict(
        self, 
        observation: Dict[str, np.ndarray],
        current_pose: np.ndarray,
        env_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Predict target pose using Mobipi optimization.
        
        Args:
            observation: Observation dict with RGB images
            current_pose: Current robot base SE2 pose [x, y, theta]
            env_info: Environment info (contains target object position)
            
        Returns:
            predicted_pose: Optimized SE2 pose
            done: Always True (one-shot predictor)
            info: Dict with optimization score, candidates, etc.
        """
        start_time = time.time()
        
        # 1. Extract target features from current observation
        target_features = self._extract_target_features(observation)
        
        # 2. Initialize candidate poses
        # Sample around target object position if available
        if env_info is not None and 'position' in env_info:
            target_pos = env_info['position'][:2]  # (x, y)
            candidates = self._sample_candidate_poses(target_pos, self.num_candidates)
        else:
            # Sample around current pose
            candidates = self._sample_candidate_poses(current_pose[:2], self.num_candidates)
        
        # 3. Run Bayesian Optimization to find best pose
        best_pose, best_score, all_scores = self._optimize_pose(
            candidates, 
            target_features,
            optimization_steps=self.optimization_steps
        )
        
        prediction_time = time.time() - start_time
        
        # 4. Build info dict
        info = {
            'score': float(best_score),
            'optimization_steps': self.optimization_steps,
            'num_candidates': self.num_candidates,
            'candidate_scores': all_scores.tolist() if isinstance(all_scores, np.ndarray) else all_scores,
            'encoder_type': self.encoder_type,
            'is_valid': True,
            'prediction_time': prediction_time
        }
        
        return best_pose, True, info
    
    def _extract_target_features(self, observation: Dict[str, np.ndarray]) -> torch.Tensor:
        """Extract target object features from observation.
        
        Args:
            observation: Observation dict with RGB images
            
        Returns:
            features: Target feature tensor
        """
        # TODO: Implement feature extraction using encoder_type
        # This requires loading the appropriate encoder (DINO, policy, etc.)
        
        # Placeholder: return random features
        return torch.randn(512).to(self.device)
    
    def _sample_candidate_poses(self, center_pos: np.ndarray, num_candidates: int) -> np.ndarray:
        """Sample candidate poses around center position.
        
        Args:
            center_pos: Center position [x, y]
            num_candidates: Number of candidates to sample
            
        Returns:
            candidates: (num_candidates, 3) array of SE2 poses
        """
        # Sample in a circle around center
        radius = 0.5  # meters
        angles = np.random.uniform(-np.pi, np.pi, num_candidates)
        distances = np.random.uniform(0, radius, num_candidates)
        
        x = center_pos[0] + distances * np.cos(angles)
        y = center_pos[1] + distances * np.sin(angles)
        theta = np.random.uniform(-np.pi, np.pi, num_candidates)
        
        candidates = np.stack([x, y, theta], axis=1)
        return candidates
    
    def _optimize_pose(
        self, 
        initial_candidates: np.ndarray,
        target_features: torch.Tensor,
        optimization_steps: int
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Optimize pose using Bayesian Optimization.
        
        Args:
            initial_candidates: (N, 3) initial candidate poses
            target_features: Target feature tensor
            optimization_steps: Number of optimization steps
            
        Returns:
            best_pose: Optimized SE2 pose
            best_score: Score of best pose
            all_scores: Scores of all candidates
        """
        if self.scene_model is None:
            raise RuntimeError("Scene model not loaded. Call load_checkpoint() first.")
        
        # Import optimization utils
        try:
            from mobipi.utils.opt_utils import optimize_pose_batch
            from mobipi.utils.score_utils import HybridDistribution
        except ImportError:
            raise ImportError("Mobipi optimization utils not found.")
        
        # Create score distribution
        dist = HybridDistribution(
            encoder=self.feature_encoder,
            device=self.device
        )
        
        # Define score function for optimization
        def score_function(robot_poses):
            # Render images from each pose using 3DGS
            # Extract features and compute similarity
            scores = []
            for pose in robot_poses:
                # Render from this pose
                rendered_images = self._render_from_pose(pose)
                # Extract features
                features = self.feature_encoder.extract(rendered_images)
                # Compute similarity score
                score = dist.compute_similarity(features, target_features)
                scores.append(score)
            return np.array(scores)
        
        # Run Bayesian Optimization
        best_pose, best_score, all_scores = optimize_pose_batch(
            score_function=score_function,
            initial_candidates=initial_candidates,
            num_iterations=optimization_steps,
            num_candidates=self.num_candidates
        )
        
        return best_pose, best_score, all_scores
    
    def _render_from_pose(self, pose: np.ndarray) -> torch.Tensor:
        """Render images from given pose using 3DGS.
        
        Args:
            pose: SE2 pose [x, y, theta]
            
        Returns:
            rendered_images: Rendered RGB images
        """
        # Convert SE2 to camera extrinsics
        # Render using BatchSceneModel
        # This requires camera intrinsics and relative camera poses
        # Simplified implementation - actual requires full camera setup
        
        # TODO: Implement full rendering pipeline
        # For now, return placeholder
        return torch.randn(1, 3, 224, 224).to(self.device)
    
    def reset(self):
        """Reset predictor state."""
        pass
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load 3DGS scene model and feature encoder.
        
        Args:
            checkpoint_path: Path to scene model directory
        """
        # Import mobipi modules
        try:
            from mobipi.scene_model.scene_model import BatchSceneModel
            from mobipi.utils.encoder_utils import (
                DinoDenseDescriptorEncoder, 
                DinoEncoder, 
                PolicyEncoder
            )
        except ImportError:
            raise ImportError(
                "Mobipi module not found. Make sure predictor/mobipi is installed."
            )
        
        # Load BatchSceneModel
        # Requires camera intrinsics - these should be set during predict()
        # For now, store the model path
        self.scene_model_path = checkpoint_path
        self.scene_model = None  # Will be initialized in predict() with camera params
        
        # Load feature encoder based on encoder_type
        if self.encoder_type == 'dino_dense_descriptor':
            self.feature_encoder = DinoDenseDescriptorEncoder(device=self.device)
        elif self.encoder_type == 'dino':
            self.feature_encoder = DinoEncoder(device=self.device)
        elif self.encoder_type == 'policy':
            # PolicyEncoder requires the manipulation policy checkpoint
            if self.policy_checkpoint_path is None:
                raise ValueError("policy_checkpoint_path required for PolicyEncoder")
            self.feature_encoder = PolicyEncoder(
                policy_checkpoint_path=self.policy_checkpoint_path,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")
    
    @property
    def name(self) -> str:
        """Return predictor name."""
        return "mobipi"
