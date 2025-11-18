"""Blank predictor - baseline that doesn't predict anything.

Simply returns the current robot pose without any prediction.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any

from benchmark.predictor.base import BasePredictor


class BlankPredictor(BasePredictor):
    """Blank baseline predictor.
    
    Does not perform any prediction - simply returns current robot pose.
    This serves as a baseline to evaluate the benefit of navigation prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize blank predictor.
        
        Args:
            config: Configuration dictionary (not used for blank predictor)
        """
        super().__init__(config)
    
    def predict(
        self, 
        observation: Dict[str, np.ndarray],
        current_pose: np.ndarray,
        env_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Return current pose without prediction.
        
        Args:
            observation: Observation dict (not used)
            current_pose: Current robot base SE2 pose [x, y, theta]
            env_info: Environment info (not used)
            
        Returns:
            predicted_pose: Same as current_pose
            done: Always True (one-shot)
            info: Empty dict
        """
        # Simply return current pose - no prediction
        predicted_pose = current_pose.copy()
        
        info = {
            'is_valid': True,
            'prediction_time': 0.0  # No computation time
        }
        
        return predicted_pose, True, info
    
    def reset(self):
        """Reset predictor state (no-op for blank predictor)."""
        pass
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint (no-op for blank predictor)."""
        pass
    
    @property
    def name(self) -> str:
        """Return predictor name."""
        return "blank"
