#!/usr/bin/env python3
"""Main benchmark runner script."""

import hydra
from omegaconf import DictConfig
import numpy as np
import os
import json

from benchmark.env.env_utils import create_env_from_config
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
        python scripts/run_benchmark.py env.name=PnPCounterToCab predictor=blank policy=bc_transformer benchmark=evaluation
        python scripts/run_benchmark.py env.name=CloseDrawer predictor=blank policy=diffusion benchmark=evaluation
    """
    assert cfg.benchmark.mode == 'evaluation', "Use benchmark=evaluation"
    
    # Suppress logging from robomimic and robosuite
    import logging
    logging.getLogger('robomimic').setLevel(logging.ERROR)
    logging.getLogger('robosuite').setLevel(logging.ERROR)
    logging.getLogger('robosuite_logs').setLevel(logging.ERROR)
    
    # Load JSON config to initialize ObsUtils (following collect_n2m_data.py)
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.config import config_factory
    
    json_config_path = os.path.expanduser(cfg.env.json_config_path)
    with open(json_config_path, 'r') as f:
        ext_cfg = json.load(f)
    
    # Save json config for later use
    json_config = ext_cfg
    
    # Create config object using config_factory
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
    
    # Initialize ObsUtils with config
    ObsUtils.initialize_obs_utils_with_config(config)
    
    # Set seed (following collect_n2m_data.py: seed * 1000)
    base_seed = config.train.seed if hasattr(config.train, 'seed') else cfg.seed
    actual_seed = base_seed * 1000
    np.random.seed(actual_seed)
    
    # Create environment (using JSON config)
    env = create_env_from_config(
        env_config=dict(cfg.env),
        seed=actual_seed,
        render=None,  # Use render setting from YAML/JSON config
        render_offscreen=None
    )
    
    # Load manipulation policy (following collect_n2m_data.py)
    print("\n============= Loading Manipulation Policy =============")
    from robomimic.algo import algo_factory, RolloutPolicy
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils
    import robomimic.utils.train_utils as TrainUtils
    import robomimic.utils.lang_utils as LangUtils
    
    # Get device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    
    # Load dataset to get shape_meta
    dataset_path = os.path.expanduser(config.train.data[0]["path"])
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        ds_format=config.train.data_format,
        verbose=True
    )
    
    # Create model
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    
    # Load checkpoint
    ckpt_path = config.experiment.ckpt_path
    if ckpt_path is not None and os.path.isfile(os.path.expanduser(ckpt_path)):
        print(f"Loading model weights from {ckpt_path}")
        ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])
    else:
        raise ValueError(f"Checkpoint path not found or not specified: {ckpt_path}")
    
    # Load training dataset to get normalization stats
    lang_encoder = LangUtils.LangEncoder(device=device)
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"], lang_encoder=lang_encoder)
    
    # Get normalization stats
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()
    action_normalization_stats = trainset.get_action_normalization_stats()
    
    # Wrap as RolloutPolicy
    rollout_policy = RolloutPolicy(
        model,
        obs_normalization_stats=obs_normalization_stats,
        action_normalization_stats=action_normalization_stats,
        lang_encoder=lang_encoder,
    )
    print("Policy loaded successfully\n")
    
    # Load predictor
    predictor = _create_predictor(cfg, env)
    
    # Get horizon from JSON config (following reference: train_utils.py:119)
    # Priority: train.data[0].horizon > experiment.rollout.horizon
    if isinstance(config.train.data, list) and len(config.train.data) > 0:
        dataset_cfg = config.train.data[0]
        if isinstance(dataset_cfg, dict) and 'horizon' in dataset_cfg:
            horizon = dataset_cfg['horizon']
        else:
            horizon = config.experiment.rollout.horizon
    else:
        horizon = config.experiment.rollout.horizon
    
    # Get render from robocasa.yaml (not from benchmark config)
    render_enabled = cfg.env.render if hasattr(cfg.env, 'render') else False
    
    # Build benchmark config
    benchmark_config = dict(cfg.benchmark)
    benchmark_config['horizon'] = horizon
    benchmark_config['render'] = render_enabled
    
    # Get policy name from config
    policy_name = cfg.policy.name if hasattr(cfg.policy, 'name') else 'bc_transformer'
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        env=env,
        policy=rollout_policy,
        predictor=predictor,
        config=benchmark_config,
        algo_name=config.algo_name,
        policy_name=policy_name
    )
    
    # Run evaluation
    results = runner.run_evaluation()
    
    # Save results
    # Get environment metadata
    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env
    
    task_name = cfg.env.name
    scene_id = unwrapped_env.layout_id
    style_id = unwrapped_env.style_id
    policy_type = cfg.policy.name if hasattr(cfg.policy, 'name') else 'bc_transformer'
    predictor_name = cfg.predictor.name
    
    # Format: {task}_{scene}_{style}_{policy}_{predictor}.json
    output_filename = f"{task_name}_{scene_id}_{style_id}_{policy_type}_{predictor_name}.json"
    output_dir = cfg.benchmark.get('results_dir', 'results/benchmark')
    output_path = os.path.join(output_dir, output_filename)
    runner.save_results(results, output_path)
    
    # Close environment
    env.close()


def _create_predictor(cfg: DictConfig, env):
    """Create predictor from config.
    
    Args:
        cfg: Hydra config
        env: Environment instance (needed for some predictors)
    
    Returns:
        predictor: Predictor instance
    """
    predictor_name = cfg.predictor.name
    
    print(f"\n============= Loading Predictor: {predictor_name} =============")
    
    if predictor_name == 'blank':
        predictor = BlankPredictor(dict(cfg.predictor))
    elif predictor_name == 'n2m':
        predictor = N2MPredictor(dict(cfg.predictor), env)
        if hasattr(cfg.predictor, 'checkpoint_path') and cfg.predictor.checkpoint_path:
            predictor.load_checkpoint(cfg.predictor.checkpoint_path)
    elif predictor_name == 'lelan':
        predictor = LeLaNPredictor(dict(cfg.predictor), env)
        if hasattr(cfg.predictor, 'checkpoint_path') and cfg.predictor.checkpoint_path:
            predictor.load_checkpoint(cfg.predictor.checkpoint_path)
    elif predictor_name == 'mobipi':
        predictor = MobipiPredictor(dict(cfg.predictor), env)
        if hasattr(cfg.predictor, 'scene_model_path') and cfg.predictor.scene_model_path:
            predictor.load_checkpoint(cfg.predictor.scene_model_path)
    elif predictor_name == 'reachability':
        predictor = ReachabilityPredictor(dict(cfg.predictor), env)
    else:
        raise ValueError(f"Unknown predictor: {predictor_name}")
    
    print(f"Predictor '{predictor_name}' loaded successfully\n")
    return predictor


if __name__ == "__main__":
    main()
