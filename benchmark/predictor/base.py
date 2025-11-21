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
    
    def __init__(self):
        """Initialize predictor."""

    @abstractmethod
    def predict(self):
        """Predict robot base target pose or pose increment."""
        pass
    
    def needs_detect_mode(self) -> bool:
        """Whether predictor needs DETECT mode (robot-mounted camera)."""
        return False
    
    def needs_robot_removal(self) -> bool:
        """Whether predictor needs robot to be moved away for scene capture."""
        return False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return predictor name."""
        pass
