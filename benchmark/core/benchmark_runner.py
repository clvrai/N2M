"""Main benchmark runner."""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
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
        config: Dict[str, Any],
        algo_name: str = "bc",
        policy_name: str = "bc_transformer"
    ):
        """Initialize benchmark runner.
        
        Args:
            env: RoboCasa environment
            policy: Manipulation policy
            predictor: Navigation predictor
            config: Benchmark configuration
            algo_name: Algorithm name ('act', 'bc', 'diffusion', etc.)
            policy_name: Policy name for results (e.g., 'bc_transformer', 'diffusion')
        """
        self.env = env
        self.policy = policy
        self.predictor = predictor
        self.config = config
        self.algo_name = algo_name
        self.policy_name = policy_name
        
    def run_evaluation(self) -> Dict[str, Any]:
        """Run benchmark evaluation.
        
        Returns:
            results: Dictionary with evaluation results
        """
        num_episodes = self.config.get('num_episodes', 50)
        
        # Get task area randomization settings for evaluation
        task_area_randomization = self.config.get('task_area_randomization', None)
        
        # Initialize collision checker if needed
        collision_checker = None
        if task_area_randomization is not None:
            # Build collision checker from first observation
            obs = self.env.reset()
            # TODO: Build collision checker from observation
            # This requires point cloud, which we'll handle in data collection
            pass
        
        # Run episodes
        episodes = []
        for episode_id in tqdm(range(num_episodes), desc="Running evaluation"):
            # Sample initial pose if using task area randomization
            initial_pose = None
            if task_area_randomization is not None and collision_checker is not None:
                initial_pose = sample_collision_free_pose(
                    collision_checker,
                    task_area_randomization,
                    origin_pose=np.array([0.0, 0.0, 0.0]),
                    max_tries=100
                )
            
            # Run rollout
            result = run_rollout_with_predictor(
                self.env,
                self.policy,
                self.predictor,
                self.config,
                algo_name=self.algo_name,
                initial_pose=initial_pose
            )
            
            # Add episode ID
            result['episode_id'] = episode_id
            episodes.append(result)
        
        # Compute statistics
        successes = [ep['success'] for ep in episodes]
        success_rate = np.mean(successes)
        
        steps = [ep['num_steps'] for ep in episodes]
        mean_steps = np.mean(steps)
        std_steps = np.std(steps)
        
        pred_times = [ep['prediction_time'] for ep in episodes]
        mean_pred_time = np.mean(pred_times)
        std_pred_time = np.std(pred_times)
        
        # Build results
        results = {
            'config': {
                'predictor': self.predictor.name,
                'policy': self.policy_name,
                'algo_name': self.algo_name,
                'task': self.config.get('task_name', 'unknown'),
                'num_episodes': num_episodes,
                'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
            },
            'episodes': episodes,
            'statistics': {
                'success_rate': float(success_rate),
                'mean_steps': float(mean_steps),
                'std_steps': float(std_steps),
                'mean_prediction_time': float(mean_pred_time),
                'std_prediction_time': float(std_pred_time)
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
