"""Base class for all manipulation policies."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BasePolicy(ABC):
    """Base class for all manipulation policies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize policy.
        
        Args:
            config: Policy configuration dictionary
        """
        self.config = config
        
    @abstractmethod
    def predict_action(
        self, 
        observation: Dict[str, np.ndarray],
        goal: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Predict action given observation.
        
        Args:
            observation: Observation dictionary (compatible with robomimic format)
            goal: Goal observation (for goal-conditioned policies)
            
        Returns:
            action: Action array
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset policy state (for RNN or stateful policies)."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        """Load policy checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return policy name."""
        pass
