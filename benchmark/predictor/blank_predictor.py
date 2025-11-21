"""Blank predictor - baseline that doesn't predict anything.

Simply returns the current robot pose without any prediction.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any

from benchmark.predictor.base import BasePredictor
from benchmark.utils.collision_utils import CollisionChecker

class BlankPredictor(BasePredictor):
    """Blank baseline predictor. 
    
    Does not perform any prediction - simply returns current robot pose.
    This serves as a baseline to evaluate the benefit of navigation prediction.
    """
    def __init__(self, hydra_cfg, json_config, env, unwrapped_env):
        super().__init__()  # BasePredictor.__init__() takes no arguments
        self.hydra_cfg = hydra_cfg
        self.json_config = json_config
        self.env = env
        self.unwrapped_env = unwrapped_env
        
    def predict(self, se2_initial, se2_randomized, collision_checker: CollisionChecker):
        """Return current pose without prediction."""
        # Simply return current pose - no prediction
        result = {
            'is_ego': False,
            'se2_predicted': se2_randomized,
        }
        return result

    # ====== overwrite base class methods ======

    def needs_detect_mode(self) -> bool:
        """Blank predictor doesn't need DETECT mode (no observation needed)."""
        return False
    
    def needs_robot_removal(self) -> bool:
        """Blank predictor doesn't need robot removal (no scene capture needed)."""
        return False
    
    @property
    def name(self) -> str:
        """Return predictor name."""
        return "blank"
