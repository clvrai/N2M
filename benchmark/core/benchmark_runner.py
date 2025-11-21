"""Main benchmark runner."""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import numpy as np

from benchmark.core.rollout import run_rollout_with_predictor
from benchmark.predictor.base import BasePredictor
from benchmark.policy.base import BasePolicy
from benchmark.utils.sampling_utils import sample_collision_free_pose
from benchmark.utils.collision_utils import CollisionChecker
from benchmark.utils.observation_utils import observation_to_pointcloud, get_camera_params


class BenchmarkRunner:
    """Main benchmark runner for evaluating predictors."""
    
    def __init__(
        self,
        env,
        policy: BasePolicy,
        predictor: BasePredictor,
        cfg: Any,
        algo_name: str = "bc",
        policy_name: str = "bc_transformer"
    ):
        """Initialize benchmark runner.
        
        Args:
            env: RoboCasa environment
            policy: Manipulation policy
            predictor: Navigation predictor
            cfg: Full Hydra config (DictConfig), provides access to cfg.benchmark, cfg.env, cfg.seed, etc.
            algo_name: Algorithm name ('act', 'bc', 'diffusion', etc.)
            policy_name: Policy name for results (e.g., 'bc_transformer', 'diffusion')
        """
        self.env = env
        self.policy = policy
        self.predictor = predictor
        self.cfg = cfg
        self.algo_name = algo_name
        self.policy_name = policy_name
        self.collision_checker = None
        
    def _build_collision_checker(self) -> CollisionChecker:
        """Build collision checker from depth cameras.
        
        Following collect_n2m_data.py implementation:
        1. Move robot away to avoid occlusion
        2. Capture point clouds from depth cameras
        3. Build collision checker from merged point cloud
        4. Restore robot position
        
        Returns:
            collision_checker: CollisionChecker instance, or None if failed
        """
        # Get unwrapped env to access simulator
        unwrapped_env = self.env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        
        # Move robot away to avoid occlusion (following collect_n2m_data.py)
        from benchmark.utils.transform_utils import qpos_command_wrapper
        robot = unwrapped_env.robots[0]
        original_qpos = unwrapped_env.sim.data.qpos[robot._ref_base_joint_pos_indexes].copy()
        unwrapped_env.sim.data.qpos[robot._ref_base_joint_pos_indexes] = qpos_command_wrapper(np.array([0.0, -50.0, 0.0]))
        unwrapped_env.sim.forward()
        ac = np.zeros(self.env.action_dimension if hasattr(self.env, 'action_dimension') else unwrapped_env.action_dim)
        self.env.step(ac)
        
        # Capture point cloud from depth cameras
        from robocasa.demos.kitchen_scenes import capture_depth_camera_data
        
        # Get depth camera names from config
        # Note: Collision checker uses global scene cameras (depth_camera1-5) to build occupancy grid
        # This is different from N2M predictor which uses robot0_front_depth from initial position
        depth_cameras = self.cfg.env.depth_cameras if hasattr(self.cfg, 'env') and hasattr(self.cfg.env, 'depth_cameras') else ['robot0_agentview_left_depth', 'robot0_agentview_right_depth', 'robot0_eye_in_hand_depth']
        
        point_clouds = []
        for cam_name in depth_cameras:
            pcd_cam = capture_depth_camera_data(unwrapped_env, camera_name=cam_name)
            if pcd_cam is not None and len(pcd_cam.points) > 0:
                point_clouds.append(pcd_cam)
        
        # Merge all point clouds
        
        if len(point_clouds) > 0:
            pcd = point_clouds[0]
            for pcd_cam in point_clouds[1:]:
                pcd += pcd_cam
            
            # Build collision checker
            self.collision_checker = CollisionChecker(
                point_cloud=pcd,
                resolution=0.02,
                robot_width=0.5,
                robot_length=0.63,
                ground_z=0.05
            )
        else:
            print("Warning: No point clouds captured, collision checker not built")
        
        # Restore robot to original position
        unwrapped_env.sim.data.qpos[robot._ref_base_joint_pos_indexes] = original_qpos
        unwrapped_env.sim.forward()
        
    def _load_existing_results(self, results_path: str) -> Tuple[List[Dict], int]:
        """Load existing results from file if available.
        
        Args:
            results_path: Path to results JSON file
            
        Returns:
            episodes: List of completed episode results
            num_completed: Number of completed episodes
        """
        import os
        if not os.path.exists(results_path):
            return [], 0
        
        try:
            import json
            with open(results_path, 'r') as f:
                existing_results = json.load(f)
            
            episodes = existing_results.get('episodes', [])
            num_completed = len(episodes)
            print(f"Found {num_completed} completed episodes in {results_path}")
            return episodes, num_completed
        except Exception as e:
            print(f"Warning: Failed to load existing results from {results_path}: {e}")
            return [], 0
    
    def _save_incremental_results(self, results: Dict[str, Any], results_path: str):
        """Save results incrementally after each episode.
        
        Args:
            results: Current results dictionary
            results_path: Path to save results
        """
        import os
        import json
        
        # Create directory if needed
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        # Save to file
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_encoder)
    
    def _json_encoder(self, obj):
        """Helper to encode numpy types for JSON."""
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return str(obj)
    
    def run_evaluation(self, results_path: str = None) -> Dict[str, Any]:
        """Run benchmark evaluation with incremental saving and resume capability.
        
        Args:
            results_path: Path to save results (for incremental saving and resume)
        
        Returns:
            results: Dictionary with evaluation results
        """
        # Get benchmark config from cfg
        num_episodes = self.cfg.benchmark.get('num_episodes', 50) if hasattr(self.cfg, 'benchmark') else 50
        
        # Get task area randomization settings for evaluation
        task_area_randomization = self.cfg.benchmark.get('task_area_randomization', None) if hasattr(self.cfg, 'benchmark') else None
        
        # Load existing results if available (for resume)
        episodes = []
        num_completed = 0
        if results_path is not None:
            episodes, num_completed = self._load_existing_results(results_path)
        
        # Check if already completed
        if num_completed >= num_episodes:
            print(f"Already completed {num_completed} episodes (>= target {num_episodes}). Nothing to evaluate.")
            # Return existing results
            return self._build_results_dict(episodes, num_episodes, num_completed)
        
        print(f"Already completed: {num_completed} episodes")
        print(f"Target: {num_episodes} episodes")
        print(f"Need to evaluate: {num_episodes - num_completed} more episodes\n")
        
        # If resuming from checkpoint, need to advance the environment RNG state
        # by doing dummy resets to match the seed sequence
        if num_completed > 0:
            print(f"Advancing environment RNG state by doing {num_completed} dummy resets...")
            for i in range(num_completed):
                self.env.reset()
            print(f"Environment RNG state synchronized to episode {num_completed}\n")
        
        # Run episodes (continue from where we left off)
        for episode_id in tqdm(range(num_completed, num_episodes), desc="Running evaluation", initial=num_completed, total=num_episodes):
            # Reset environment for new episode
            obs_dict = self.env.reset()
            
            # Get initial SE2 pose after reset
            from benchmark.utils.obs_utils import obs_to_SE2
            se2_initial = obs_to_SE2(obs_dict, algorithm_name=self.algo_name)
            
            # Build collision checker and sample randomized pose if using randomization
            se2_randomized = None
            if task_area_randomization is not None:
                self._build_collision_checker()
                
                # Get furniture position for visibility check (following collect_n2m_data.py line 266-267)
                furniture_pos = None
                unwrapped_env = self.env
                while hasattr(unwrapped_env, 'env'):
                    unwrapped_env = unwrapped_env.env
                
                if hasattr(unwrapped_env, 'init_robot_base_pos') and hasattr(unwrapped_env, 'fixtures'):
                    try:
                        furniture_name = unwrapped_env.init_robot_base_pos.name
                        furniture_pos = unwrapped_env.fixtures[furniture_name].pos[:2]
                        # print(f"Episode {episode_id}: Furniture position {furniture_pos} for visibility check")
                    except Exception as e:
                        print(f"Episode {episode_id}: Could not get furniture position: {e}")
                
                # Sample from se2_initial as origin (not [0,0,0])
                se2_randomized = sample_collision_free_pose(
                    self.collision_checker,
                    task_area_randomization,
                    se2_initial=se2_initial,  # Use actual robot pose after reset as origin
                    max_tries=100,
                    visualize=False,
                    save_path=f"debug/episode_{episode_id}_sampling.png",
                    object_pos=furniture_pos,
                    check_visibility=True,
                    check_boundary=True
                )
            
            # Run rollout
            rollout_config = {}
            if hasattr(self.cfg, 'benchmark'):
                rollout_config = dict(self.cfg.benchmark)
            
            print("[DEBUG benchmark] se2_initial: ", se2_initial)
            print("[DEBUG benchmark] se2_randomized: ", se2_randomized)

            result = run_rollout_with_predictor(
                self.env,
                self.policy,
                self.predictor,
                rollout_config,
                collision_checker=self.collision_checker,
                algo_name=self.algo_name,
                se2_initial=se2_initial,
                se2_randomized=se2_randomized
            )
            
            # Add episode ID
            result['episode_id'] = episode_id
            episodes.append(result)
        
            # Incremental save after each episode (following collect_n2m_data.py pattern)
            if results_path is not None:
                current_results = self._build_results_dict(episodes, num_episodes, len(episodes))
                self._save_incremental_results(current_results, results_path)
                print(f"Episode {episode_id}: SUCCESS={result['success']}, saved to {results_path}")
        
        # Build final results
        results = self._build_results_dict(episodes, num_episodes, len(episodes))
        
        # Final save (redundant but safe)
        if results_path is not None:
            self._save_incremental_results(results, results_path)
        
        return results
    
    def _build_results_dict(self, episodes: List[Dict], num_episodes: int, num_completed: int) -> Dict[str, Any]:
        """Build results dictionary from episodes.
        
        Args:
            episodes: List of episode results
            num_episodes: Target number of episodes
            num_completed: Number of completed episodes
            
        Returns:
            results: Dictionary with evaluation results
        """
        import time
        import numpy as np
        
        # Compute statistics from completed episodes
        if len(episodes) > 0:
            successes = [ep['success'] for ep in episodes]
            success_rate = np.mean(successes)
            
            steps = [ep['num_steps'] for ep in episodes]
            mean_steps = np.mean(steps)
            std_steps = np.std(steps)
            
            pred_times = [ep['prediction_time'] for ep in episodes]
            mean_pred_time = np.mean(pred_times)
            std_pred_time = np.std(pred_times)
            
            manip_times = [ep['manipulation_time'] for ep in episodes]
            mean_manip_time = np.mean(manip_times)
            std_manip_time = np.std(manip_times)
        else:
            success_rate = 0.0
            mean_steps = 0.0
            std_steps = 0.0
            mean_pred_time = 0.0
            std_pred_time = 0.0
            mean_manip_time = 0.0
            std_manip_time = 0.0
        
        # Get seed from cfg if available
        seed = self.cfg.benchmark.get('seed', None) if hasattr(self.cfg, 'benchmark') else None
        
        # Get scene and style from environment
        scene_id = None
        style_id = None
        unwrapped_env = self.env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        
        if hasattr(unwrapped_env, 'layout_id'):
            scene_id = unwrapped_env.layout_id
        if hasattr(unwrapped_env, 'style_id'):
            style_id = unwrapped_env.style_id
        
        results = {
            'config': {
                'predictor': self.predictor.name,
                'policy': self.policy_name,
                'task': self.env.name,
                'seed': seed,
                'scene': scene_id,
                'style': style_id,
                'num_episodes': num_episodes,
                'num_completed': num_completed,
                'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
            },
            'episodes': episodes,
            'statistics': {
                'success_rate': float(success_rate),
                'mean_steps': float(mean_steps),
                'std_steps': float(std_steps),
                'mean_prediction_time': float(mean_pred_time),
                'std_prediction_time': float(std_pred_time),
                'mean_manipulation_time': float(mean_manip_time),
                'std_manipulation_time': float(std_manip_time),
                'num_success': int(np.sum(successes)) if len(episodes) > 0 else 0
            }
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file.
        
        Args:
            results: Results dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        print(f"Success rate: {results['statistics']['success_rate']:.2%}")
