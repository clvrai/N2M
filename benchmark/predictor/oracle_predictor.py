"""Oracle predictor - upper-bound baseline that returns the training-distribution pose.

Returns `se2_initial` (the unperturbed pose from env.reset()), which matches the
pose distribution the manipulation policy was trained on. Useful for isolating
manipulation difficulty from navigation difficulty.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any

from benchmark.predictor.base import BasePredictor
from benchmark.utils.collision_utils import CollisionChecker

class OraclePredictor(BasePredictor):
    """Oracle baseline predictor.

    Does not perform any prediction - returns the canonical training-time pose
    (`se2_initial`). Serves as an upper-bound baseline that isolates manipulation
    difficulty from navigation difficulty.
    """
    def __init__(self, hydra_cfg, json_config, env, unwrapped_env):
        super().__init__()  # BasePredictor.__init__() takes no arguments
        self.hydra_cfg = hydra_cfg
        self.json_config = json_config
        self.env = env
        self.unwrapped_env = unwrapped_env

    def predict(self, se2_initial, se2_randomized, collision_checker: CollisionChecker, episode_id=None):
        """Return the training-distribution pose without any prediction."""
        # Historically returned se2_randomized; current behavior returns se2_initial.
        # result = {
        #     'is_ego': False,
        #     'se2_predicted': se2_randomized,
        # }
        result = {
            'is_ego': False,
            'se2_predicted': se2_initial,
        }
        return result

    # ====== overwrite base class methods ======

    def needs_detect_mode(self) -> bool:
        """Oracle predictor doesn't need DETECT mode (no observation needed)."""
        return False

    def needs_robot_removal(self) -> bool:
        """Oracle predictor doesn't need robot removal (no scene capture needed)."""
        return False

    @property
    def name(self) -> str:
        """Return predictor name."""
        return "oracle"
