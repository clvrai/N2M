"""Base class for all predictors."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import numpy as np


class BasePredictor(ABC):
    """Base class for all predictors.
    
    All predictors use a unified iterative interface:
    - One-shot predictors (N2M, mobipi, blank): return done=True on first predict() call
    - Iterative predictors (lelan, reachability): return done=False until completion
    
    The benchmark will loop calling predict() until done=True is returned.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize predictor.
        
        Args:
            config: Predictor configuration dictionary
        """
        self.config = config
        
    @abstractmethod
    def predict(
        self, 
        observation: Dict[str, np.ndarray],
        current_pose: np.ndarray,
        env_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Predict robot base target pose or pose increment.
        
        Args:
            observation: Observation dict containing RGB, depth, etc.
            current_pose: Current robot base SE2 pose [x, y, theta]
            env_info: Additional environment information (e.g., target object position)
            
        Returns:
            predicted_pose: shape (3,) pose [x, y, theta]
                - One-shot predictor: directly return final target pose
                - Iterative predictor: can return pose delta or next step pose
            done: Whether prediction is complete
                - True: prediction complete, can start manipulation
                - False: need to continue calling predict()
            info: Dictionary with additional information:
                - 'distribution': predicted distribution (for N2M)
                - 'score': prediction confidence (for mobipi)
                - 'samples': sampled candidate poses (for N2M)
                - 'is_valid': whether prediction is valid
                - 'control': control command (for lelan) [linear_x, angular_z]
                - 'is_pose_delta': bool, True if returned pose is a delta
                - 'prediction_time': prediction time in seconds
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset predictor state.
        
        Called at the beginning of each episode:
        - One-shot predictor: can be no-op
        - Iterative predictor: reset internal counters, states, etc.
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        """Load pretrained model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return predictor name."""
        pass
