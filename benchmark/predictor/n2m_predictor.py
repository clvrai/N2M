"""N2M predictor - predicts poses from point cloud using GMM.

Wraps N2Mmodule with path resolution and point cloud capture.
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional, Any
import torch
from omegaconf import DictConfig
from benchmark.predictor.base import BasePredictor
from n2m.module.N2Mmodule import N2Mmodule
import os
from robocasa.demos.kitchen_scenes import capture_depth_camera_data
from benchmark.utils.observation_utils import pcd_global_to_local
from benchmark.utils.sample_utils import arm_fake_controller
from benchmark.utils.collision_utils import CollisionChecker

class N2MPredictor(BasePredictor):
    """N2M predictor using point cloud → GMM → sampled pose.
    
    One-shot predictor that returns done=True on first call.
    """
    
    def __init__(self, hydra_cfg: DictConfig, json_config, env, unwrapped_env):
        """Initialize N2M predictor.
        
        Args:
            hydra_cfg: Hydra config
            json_config: Robomimic/Robocasa config
            env: Environment instance (step)
            unwrapped_env: Unwrapped environment instance (forward)
                - config_path_template: Path template for config.json with {base_dir}, {task}, {layout}, {style}, {policy}
                - ckpt_path_template: Path template for checkpoint .pth file
                - camera_names: List of camera names for depth observation
            env: Environment instance (needed to extract layout/style for path construction)
        """
        super().__init__()  # Call parent class __init__
        
        self.hydra_cfg = hydra_cfg
        self.json_config = json_config
        self.env = env
        self.unwrapped_env = unwrapped_env

        self.n2m_model = None
        self.camera_name = hydra_cfg.predictor.camera_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_checkpoint()
        
        
    def predict(self, se2_initial, se2_randomized, collision_checker: CollisionChecker):
        """Predict target pose using N2M.
        
        Args:
            se2_initial: Initial SE2 pose of robot after reset [x, y, theta]
            se2_randomized: Randomized SE2 pose for task area randomization [x, y, theta]
            collision_checker: Collision checker instance
            
        Returns:
            predicted_pose: Sampled SE2 pose from GMM
            done: Always True (one-shot predictor)
            info: Dict with prediction metadata
        """
        
        pcd_global = capture_depth_camera_data(self.unwrapped_env, camera_name=self.camera_name)
        pcd_global_numpy = np.concatenate([pcd_global.points, pcd_global.colors], axis=1)

        # Apply robot_centric transformation if needed
        pcd_ego_numpy = pcd_global_to_local(pcd_global_numpy, se2_randomized)

        se2_predicted_ego, extra_info = self.n2m_model.predict(pcd_ego_numpy, collision_checker=collision_checker)

        # Simply return current pose - no prediction
        result = {
            'is_ego': True,
            'se2_predicted': se2_predicted_ego,
            'extra_info': extra_info
        }
        return result

    def reset(self):
        """Reset predictor state."""
        # Collision checker can be reused across episodes
        pass
    
    def needs_detect_mode(self) -> bool:
        """N2M needs DETECT mode to position robot-mounted camera correctly.
        
        Following train_utils.py:909, N2M uses robot0_front_depth camera
        which requires the robot arm to be in DETECT pose.
        """
        return True
    
    def needs_robot_removal(self) -> bool:
        """N2M doesn't need robot removal during evaluation.
        
        Unlike data collection (which moves robot away to capture global scene),
        evaluation uses the robot's current view from robot0_front_depth.
        """
        return False
    
    def _resolve_path_template(self, path_template: str, task: str, layout: int, style: int, policy: str) -> str:
        """Resolve path template with base_dir/task/layout/style/policy placeholders.
        
        Args:
            path_template: Path string with {base_dir}, {task}, {layout}, {style}, {policy} placeholders
            task: Task name (e.g., "OpenSingleDoor")
            layout: Layout ID (e.g., 0)
            style: Style ID (e.g., 1)
            policy: Policy name (e.g., "diffusion")
            
        Returns:
            resolved_path: Path with placeholders replaced
        """
        resolved = path_template.format(
            base_dir=self.base_dir,
            task=task,
            layout=layout,
            style=style,
            policy=policy
        )
        return resolved
    
    def load_checkpoint(self):
        """Load N2M model checkpoint using path templates from config."""
        
        base_dir = self.hydra_cfg.get('base_dir', 'data/predictor/n2m')
        task = self.hydra_cfg.env.name
        policy = self.hydra_cfg.policy.name
        config_path_template = self.hydra_cfg.predictor.config_path_template
        ckpt_path_template = self.hydra_cfg.predictor.ckpt_path_template
        layout = self.unwrapped_env.layout_id
        style = self.unwrapped_env.style_id
        
        print(f"\n============= N2M Predictor Path Setup =============")
        config_path = config_path_template.format(
            base_dir=base_dir,
            task=task,
            layout=layout,
            style=style,
            policy=policy
        )
        ckpt_path = ckpt_path_template.format(
            base_dir=base_dir,
            task=task,
            layout=layout,
            style=style,
            policy=policy
        )
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"N2M config not found: {config_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"N2M checkpoint not found: {ckpt_path}")

        print(f"[predictor] Resolved config_path: {config_path}")
        print(f"[predictor] Resolved ckpt_path: {ckpt_path}")
        
        # Load config from JSON
        import json
        with open(config_path, 'r') as f:
            full_config = json.load(f)
        
        n2m_config = {
            "n2mnet": full_config["n2mnet"],
            "ckpt": str(ckpt_path)  # Set checkpoint path
        }

        # Add preprocess and postprocess settings from n2mmodule section
        if "n2mmodule" in full_config:
            n2m_module_cfg = full_config["n2mmodule"]
            if "preprocess" in n2m_module_cfg:
                n2m_config["preprocess"] = n2m_module_cfg["preprocess"]
            if "postprocess" in n2m_module_cfg:
                n2m_config["postprocess"] = n2m_module_cfg["postprocess"]
        
        # Initialize N2M module with config
        # Following reference: 1_data_collection_with_rollout.py line 375
        self.n2m_model = N2Mmodule(n2m_config)
        self.n2m_model.model.to(self.device)  # N2Mmodule has .model attribute
        self.n2m_model.model.eval()
        
        print(f"[predictor] N2M model loaded successfully\n")
    
    @property
    def name(self) -> str:
        """Return predictor name."""
        return "n2m"
