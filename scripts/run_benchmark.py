#!/usr/bin/env python3
"""Main benchmark runner script."""

import hydra
from omegaconf import DictConfig
import numpy as np

from benchmark.env.env_utils import create_env_from_config
from benchmark.policy.robomimic_policy import RobomimicPolicy
from benchmark.predictor.blank_predictor import BlankPredictor
from benchmark.predictor.n2m_predictor import N2MPredictor
from benchmark.predictor.lelan_predictor import LeLaNPredictor
from benchmark.predictor.mobipi_predictor import MobipiPredictor
from benchmark.predictor.reachability_predictor import ReachabilityPredictor
from benchmark.core.benchmark_runner import BenchmarkRunner


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Run benchmark evaluation.
    
    Example usage:
        python scripts/run_benchmark.py env=PnPCounterToCab predictor=n2m policy=robomimic_bc
        python scripts/run_benchmark.py env=CloseDrawer predictor=blank
        python scripts/run_benchmark.py env=OpenSingleDoor predictor=mobipi
    """
    # Set seed
    np.random.seed(cfg.seed)
    
    # Create environment
    env = create_env_from_config(
        env_config=dict(cfg.env),
        obs_keys=cfg.env.obs_keys,
        seed=cfg.seed,
        render=cfg.benchmark.get('render', False),
        render_offscreen=cfg.benchmark.get('render_offscreen', True)
    )
    
    # Load policy
    policy = _create_policy(cfg)
    
    # Load predictor
    predictor = _create_predictor(cfg)
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        env=env,
        policy=policy,
        predictor=predictor,
        config=dict(cfg.benchmark)
    )
    
    # Run evaluation
    results = runner.run_evaluation()
    
    # Save results
    output_filename = cfg.benchmark.save_format.format(
        predictor=cfg.predictor.name,
        task=cfg.task_name,
        scene="all",
        style="all"
    )
    runner.save_results(results, output_filename)
    
    # Close environment
    env.close()


def _create_policy(cfg: DictConfig):
    """Create policy from config."""
    policy_type = cfg.policy.type
    
    if policy_type == 'robomimic':
        policy = RobomimicPolicy(dict(cfg.policy))
        policy.load_checkpoint(cfg.policy.checkpoint_path)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    return policy


def _create_predictor(cfg: DictConfig):
    """Create predictor from config."""
    predictor_type = cfg.predictor.type
    
    if predictor_type == 'blank':
        predictor = BlankPredictor(dict(cfg.predictor))
    elif predictor_type == 'n2m':
        predictor = N2MPredictor(dict(cfg.predictor))
        predictor.load_checkpoint(cfg.predictor.checkpoint_path)
    elif predictor_type == 'lelan':
        predictor = LeLaNPredictor(dict(cfg.predictor))
        predictor.load_checkpoint(cfg.predictor.checkpoint_path)
    elif predictor_type == 'mobipi':
        predictor = MobipiPredictor(dict(cfg.predictor))
        predictor.load_checkpoint(cfg.predictor.scene_model_path)
    elif predictor_type == 'reachability':
        predictor = ReachabilityPredictor(dict(cfg.predictor))
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")
    
    return predictor


if __name__ == "__main__":
    main()
