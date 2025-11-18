"""Robomimic policy wrapper.

Wraps robomimic policies to implement BasePolicy interface.
"""

import numpy as np
from typing import Dict, Any, Optional

from benchmark.policy.base import BasePolicy
import robomimic.utils.file_utils as FileUtils
from robomimic.algo import algo_factory, RolloutPolicy


class RobomimicPolicy(BasePolicy):
    """Wrapper for robomimic manipulation policies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize robomimic policy.
        
        Args:
            config: Policy configuration dictionary with keys:
                - robomimic_config_path: Path to robomimic JSON config
                - checkpoint_path: Path to policy checkpoint
                - obs_keys: List of observation keys
        """
        super().__init__(config)
        
        # Load robomimic config from JSON
        robomimic_config_path = config['robomimic_config_path']
        self.robomimic_config = FileUtils.config_from_json(robomimic_config_path)
        
        # Extract shape metadata from config
        # In actual usage, this should come from dataset, but for rollout we can use config
        self.obs_keys = config.get('obs_keys', [])
        
        # Create model
        # Note: We need shape_meta to create the model properly
        # For now, we'll create it when checkpoint is loaded
        self.model = None
        self.rollout_policy = None
        
    def predict_action(
        self, 
        observation: Dict[str, np.ndarray],
        goal: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Predict action given observation.
        
        Args:
            observation: Observation dictionary
            goal: Goal observation (for goal-conditioned policies)
            
        Returns:
            action: Action array
        """
        if self.rollout_policy is None:
            raise RuntimeError("Policy checkpoint not loaded. Call load_checkpoint() first.")
        
        return self.rollout_policy(observation, goal)
    
    def reset(self):
        """Reset policy state."""
        if self.rollout_policy is not None:
            self.rollout_policy.start_episode()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load policy checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file (.pth)
        """
        # Load checkpoint
        ckpt_dict = FileUtils.maybe_dict_from_checkpoint(checkpoint_path)
        
        # Get shape metadata from checkpoint
        if 'shape_metadata' in ckpt_dict:
            shape_meta = ckpt_dict['shape_metadata']
        else:
            # Fallback: try to infer from config
            # This is a simplified version - in practice might need dataset
            raise ValueError("Checkpoint does not contain shape_metadata")
        
        # Create model if not already created
        if self.model is None:
            import torch
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            self.model = algo_factory(
                algo_name=self.robomimic_config.algo_name,
                config=self.robomimic_config,
                obs_key_shapes=shape_meta["all_shapes"],
                ac_dim=shape_meta["ac_dim"],
                device=device,
            )
        
        # Load model weights
        self.model.deserialize(ckpt_dict["model"])
        self.model.eval()
        
        # Create rollout policy wrapper
        self.rollout_policy = RolloutPolicy(
            self.model,
            obs_normalization_stats=ckpt_dict.get("obs_normalization_stats", None)
        )
    
    @property
    def name(self) -> str:
        """Return policy name."""
        return f"robomimic_{self.robomimic_config.algo_name}"
