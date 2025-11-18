"""LeLaN predictor - language-conditioned end-to-end navigation.

Iterative predictor that outputs control commands.
Reference: N2M-benchmark/predictor/mobipi/README.md (run_lelan_in_env)
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional, Any
import torch

from benchmark.predictor.base import BasePredictor


class LeLaNPredictor(BasePredictor):
    """LeLaN end-to-end navigation predictor.
    
    Iterative predictor - outputs control commands until reaching target.
    Returns done=True when:
    1. Within distance threshold of target
    2. Max steps exceeded
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LeLaN predictor.
        
        Args:
            config: Configuration dictionary with:
                - checkpoint_path: Path to LeLaN checkpoint
                - task_description: Language task description
                - max_steps: Maximum navigation steps (default 500)
                - goal_distance_threshold: Distance threshold for done (default 0.1)
                - control_scale: Scale for control commands (default 1.0)
        """
        super().__init__(config)
        
        self.task_description = config.get('task_description', 'navigate to the target')
        self.max_steps = config.get('max_steps', 500)
        self.goal_distance_threshold = config.get('goal_distance_threshold', 0.1)
        self.control_scale = config.get('control_scale', 1.0)
        
        # LeLaN model (loaded in load_checkpoint)
        self.lelan_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Internal state for iteration
        self.step_count = 0
        self.goal_position = None
        
    def predict(
        self, 
        observation: Dict[str, np.ndarray],
        current_pose: np.ndarray,
        env_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Predict next control command / pose increment.
        
        Args:
            observation: Observation dict with RGB images
            current_pose: Current robot base SE2 pose [x, y, theta]
            env_info: Environment info (contains target object position)
            
        Returns:
            pose_delta: Relative pose change [dx, dy, dtheta] OR control [vx, wz, 0]
            done: True if reached goal or max steps
            info: Dict with 'is_pose_delta', 'control', etc.
        """
        start_time = time.time()
        
        # Get goal position from env_info (only on first call)
        if self.goal_position is None and env_info is not None:
            self.goal_position = env_info.get('position', np.array([0.0, 0.0, 1.0]))
        
        # Check if reached goal
        if self.goal_position is not None:
            distance_to_goal = np.linalg.norm(current_pose[:2] - self.goal_position[:2])
            if distance_to_goal < self.goal_distance_threshold:
                # Reached goal
                return np.zeros(3), True, {
                    'reached_goal': True,
                    'distance_to_goal': float(distance_to_goal),
                    'step_count': self.step_count
                }
        
        # Check max steps
        if self.step_count >= self.max_steps:
            return np.zeros(3), True, {
                'reached_goal': False,
                'max_steps_exceeded': True,
                'step_count': self.step_count
            }
        
        # Run LeLaN inference to get control command
        control_command = self._run_lelan_inference(observation, current_pose, self.task_description)
        
        # Convert control [linear_x, angular_z] to pose delta
        # Assume dt = 0.1 seconds per step
        dt = 0.1
        linear_x, angular_z = control_command[0], control_command[1]
        
        # SE2 kinematics: 
        # dx = linear_x * cos(theta) * dt
        # dy = linear_x * sin(theta) * dt
        # dtheta = angular_z * dt
        theta = current_pose[2]
        dx = linear_x * np.cos(theta) * dt * self.control_scale
        dy = linear_x * np.sin(theta) * dt * self.control_scale
        dtheta = angular_z * dt * self.control_scale
        
        pose_delta = np.array([dx, dy, dtheta])
        
        self.step_count += 1
        prediction_time = time.time() - start_time
        
        info = {
            'is_pose_delta': True,  # Important: tells rollout this is a delta
            'control': control_command.tolist(),
            'step_count': self.step_count,
            'distance_to_goal': float(distance_to_goal) if self.goal_position is not None else None,
            'prediction_time': prediction_time
        }
        
        return pose_delta, False, info  # done=False to continue
    
    def _run_lelan_inference(
        self, 
        observation: Dict[str, np.ndarray],
        current_pose: np.ndarray,
        task_description: str
    ) -> np.ndarray:
        """Run LeLaN model inference.
        
        Args:
            observation: Observation dict with RGB images
            current_pose: Current SE2 pose
            task_description: Language description
            
        Returns:
            control: [linear_x, angular_z] control command
        """
        if self.lelan_model is None:
            raise RuntimeError("LeLaN model not loaded. Call load_checkpoint() first.")
        
        # Get RGB image from observation
        # LeLaN expects resized 224x224 image
        import cv2
        camera_key = 'camera' + '_image'
        if camera_key in observation:
            image = observation[camera_key]
            # Handle different image shapes
            if len(image.shape) == 4:
                image = image[-1]  # Take last frame if batch
            if image.shape[0] == 3:  # CHW format
                image = image.transpose(1, 2, 0)  # Convert to HWC
            # Resize to 224x224
            image = cv2.resize(image, (224, 224))
        else:
            raise KeyError(f"Camera {camera_key} not found in observation")

        # Run LeLaN forward pass
        lelan_action = self.lelan_model.forward(image, task_description)

        # Extract control commands
        linear_x = lelan_action.linear.x
        angular_z = lelan_action.angular.z

        return np.array([linear_x, angular_z])
    
    def reset(self):
        """Reset internal state."""
        self.step_count = 0
        self.goal_position = None
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load LeLaN model checkpoint.
        
        Args:
            checkpoint_path: Path to LeLaN checkpoint
        """
        # Import LeLaN model
        try:
            # LeLaN module structure depends on predictor/lelan/train
            # This is a wrapper that should provide a forward() method
            from lelan.nav_model import LeLaNModel  # Adjust import based on actual structure
        except ImportError:
            raise ImportError(
                "LeLaN module not found. Make sure predictor/lelan is installed: "
                "pip install -e predictor/lelan/train"
            )
        
        # Load model
        self.lelan_model = LeLaNModel.load_from_checkpoint(
            checkpoint_path,
            device=self.device
        )
        self.lelan_model.eval()
    
    @property
    def name(self) -> str:
        """Return predictor name."""
        return "lelan"
