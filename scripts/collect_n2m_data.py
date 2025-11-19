#!/usr/bin/env python3
"""Collect training data for N2M predictor."""

# # Suppress annoying warnings
# import warnings
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.filterwarnings('ignore', category=UserWarning)

import hydra
from omegaconf import DictConfig
import numpy as np
from tqdm import tqdm

from benchmark.env.env_utils import create_env_from_config
from benchmark.core.data_collector import N2MDataCollector
from benchmark.utils.sample_utils import TargetHelper
from benchmark.utils.obs_utils import obs_to_SE2
import open3d as o3d


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Collect N2M training data.
    
    Example usage:
        python scripts/collect_n2m_data.py env.name=CloseDrawer policy=bc_transformer benchmark=collection benchmark.num_valid_data=20
        python scripts/collect_n2m_data.py env.name=PnPCounterToCab policy=diffusion benchmark=collection
    """
    assert cfg.benchmark.mode == 'data_collection', "Use benchmark=collection"
    assert cfg.benchmark.collect_for == 'n2m', "This script is for N2M data collection"
    
    # Suppress logging from robomimic and robosuite
    import logging
    logging.getLogger('robomimic').setLevel(logging.ERROR)
    logging.getLogger('robosuite').setLevel(logging.ERROR)
    logging.getLogger('robosuite_logs').setLevel(logging.ERROR)
    
    # Load JSON config to initialize ObsUtils (following reference implementation)
    import json
    import os
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.config import config_factory
    
    json_config_path = os.path.expanduser(cfg.env.json_config_path)
    with open(json_config_path, 'r') as f:
        ext_cfg = json.load(f)
    
    # Save json config for later use (render settings, etc.)
    json_config = ext_cfg
    
    # Create config object using config_factory (following reference implementation)
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
    
    # Initialize ObsUtils with config (following reference implementation)
    ObsUtils.initialize_obs_utils_with_config(config)
    
    # Set seed (following reference implementation: seed * 1000)
    # Reference: A_ref/N2M-sim/scripts/1_data_collection_with_rollout.py line 132
    base_seed = config.train.seed if hasattr(config.train, 'seed') else cfg.seed
    actual_seed = base_seed * 1000  # Same as reference implementation
    np.random.seed(actual_seed)
    
    # Create environment (using JSON config from A_ref/N2M-sim)
    # Following reference implementation exactly: use config.experiment.render/render_video
    env = create_env_from_config(
        env_config=dict(cfg.env),
        seed=actual_seed,  # Use same seed calculation as reference
        render=None,  # Use render setting from JSON config (experiment.render)
        render_offscreen=None  # Use setting from JSON
    )
    
    # Get env metadata for folder naming: {task}_{layoutid}_{styleid}_{policytype}
    # After wrap_env_from_config, need to unwrap to get to base RoboCasa env
    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env
    
    task_name = cfg.env.name  # e.g., PnPCounterToCab
    layout_id = unwrapped_env.layout_id
    style_id = unwrapped_env.style_id
    
    # Get policy type from config (e.g., bc_transformer, diffusion)
    policy_type = cfg.policy.name if hasattr(cfg.policy, 'name') else 'bc_transformer'
    
    # Dataset naming: {task}_{layout}_{style}_{policy}
    dataset_name = f"{task_name}_{layout_id}_{style_id}_{policy_type}"
    
    # Update save directory to use dataset naming convention: data/predictor/n2m/{task}_{layout}_{style}_{policy}/
    base_dir = "data/predictor/n2m"
    save_dir = os.path.join(base_dir, dataset_name)
    print(f"Data will be saved to: {save_dir}")
    
    # Create data collector
    collector = N2MDataCollector(output_dir=save_dir)
    
    # Check if rendering is enabled
    # Priority: YAML config > JSON config
    if 'render' in cfg.env:
        render_enabled = cfg.env.render
        print(f"[INFO] Using render setting from YAML config: {render_enabled}")
    else:
        render_enabled = json_config.get('experiment', {}).get('render', False)
        print(f"[INFO] Using render setting from JSON config: {render_enabled}")
    
    # Get depth camera names from benchmark config
    depth_cameras = cfg.benchmark.depth_cameras
    
    # Load manipulation policy (following reference implementation)
    print("\n============= Loading Manipulation Policy =============")
    from robomimic.algo import algo_factory, RolloutPolicy
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils
    
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
    
    # Load training dataset to get normalization stats (following reference: 1_data_collection_with_rollout.py:198-215)
    import robomimic.utils.lang_utils as LangUtils
    lang_encoder = LangUtils.LangEncoder(device=device)
    
    # Load training dataset (following reference: 1_data_collection_with_rollout.py:198-199)
    import robomimic.utils.train_utils as TrainUtils
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"], lang_encoder=lang_encoder)
    
    # Get normalization stats (following reference: 1_data_collection_with_rollout.py:209-215)
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()
    
    # Always get action normalization stats (following reference: 1_data_collection_with_rollout.py:215)
    action_normalization_stats = trainset.get_action_normalization_stats()
    
    # Wrap as RolloutPolicy (following reference: 1_data_collection_with_rollout.py:343-348)
    rollout_policy = RolloutPolicy(
        model,
        obs_normalization_stats=obs_normalization_stats,
        action_normalization_stats=action_normalization_stats,
        lang_encoder=lang_encoder,
    )
    print("Policy loaded successfully\n")
    
    # Collect episodes (following reference implementation: train_utils.py:836-1134)
    # Get number of already collected episodes (for incremental collection)
    num_already_collected = collector.get_num_collected()
    num_valid_data_target = cfg.benchmark.num_valid_data
    
    print(f"Already collected: {num_already_collected} episodes")
    print(f"Target: {num_valid_data_target} episodes")
    print(f"Need to collect: {max(0, num_valid_data_target - num_already_collected)} more episodes\n")
    
    if num_already_collected >= num_valid_data_target:
        print(f"Already have {num_already_collected} episodes (>= target {num_valid_data_target}). Nothing to collect.")
        env.close()
        return
    
    # Loop until we collect enough valid data
    episode_attempt = 0
    successful_episodes = num_already_collected  # Continue from where we left off
    
    with tqdm(total=num_valid_data_target, initial=num_already_collected, desc="Collecting N2M data") as pbar:
        while successful_episodes < num_valid_data_target:
            episode_attempt += 1
            
            # STEP 3: reset environment (following reference: train_utils.py:841)
            obs_dict = env.reset()
            
            # Get unwrapped env (following reference: train_utils.py:845-849)
            # Different policies have different wrapper layers
            is_act_policy = config.algo_name == "act"
            if is_act_policy:
                unwrapped_env = env.env
            else:
                unwrapped_env = env.env.env
            robot = unwrapped_env.robots[0]
            
            # Get origin SE2 pose (following reference: train_utils.py:850)
            se2_origin = obs_to_SE2(obs_dict, algorithm_name=config.algo_name)
            ac = np.zeros(12)
            
            # Render to screen if enabled
            if render_enabled and hasattr(unwrapped_env, 'viewer') and unwrapped_env.viewer is not None:
                unwrapped_env.render()
                env.step(ac)
            
            # STEP 3-2: move robot away (following reference: train_utils.py:862-864)
            from benchmark.utils.transform_utils import qpos_command_wrapper
            unwrapped_env.sim.data.qpos[robot._ref_base_joint_pos_indexes] = qpos_command_wrapper(np.array([0.0, -50.0, 0.0]))
            unwrapped_env.sim.forward()
            env.step(ac)
            
            # STEP 3-3: capture PCL (following reference: train_utils.py:880-887)
            from robocasa.demos.kitchen_scenes import capture_depth_camera_data
            
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
            else:
                print(f"Warning: No point clouds captured for attempt {episode_attempt}")
                continue
            
            # STEP 3-4: determine target_se2 (following reference: train_utils.py:890-893)
            # Use N2M data collection randomization range from benchmark config
            pose_range = cfg.benchmark.n2m_data_collection_randomization
            x_range = pose_range.x
            y_range = pose_range.y
            theta_range = pose_range.theta
            
            # Calculate half ranges
            x_half_range = (x_range[1] - x_range[0]) / 2.0
            y_half_range = (y_range[1] - y_range[0]) / 2.0
            theta_half_range_rad = (theta_range[1] - theta_range[0]) / 2.0
            theta_half_range_deg = np.degrees(theta_half_range_rad)
            
            target_helper = TargetHelper(
                pcd=pcd,
                origin_se2=se2_origin,
                x_half_range=x_half_range,
                y_half_range=y_half_range,
                theta_half_range_deg=theta_half_range_deg,
                vis=False,
                camera_intrinsic=None,
                filter_noise=True
            )
            furniture_name = unwrapped_env.init_robot_base_pos.name
            furniture_pos = unwrapped_env.fixtures[furniture_name].pos[:2]
            
            # This returns se2_delta (relative position)
            try:
                target_se2_delta = target_helper.get_random_target_se2_with_reachability_check(furniture_pos)
            except Exception as e:
                print(f"Attempt {episode_attempt}: Failed to sample pose - {e}")
                continue
            
            # Teleport robot to target pose (following reference: train_utils.py:898-902)
            from benchmark.utils.transform_utils import qpos_command_wrapper
            robot = unwrapped_env.robots[0]
            unwrapped_env.sim.data.qpos[robot._ref_base_joint_pos_indexes] = qpos_command_wrapper(target_se2_delta)
            unwrapped_env.sim.forward()
            
            # Wait for robot to stabilize (following reference: train_utils.py:1055-1057)
            for _ in range(5):
                ac = np.zeros(env.action_spec()[0].shape[0] if hasattr(env, 'action_spec') else unwrapped_env.action_dim)
                obs_dict, _, _, _ = env.step(ac)
            
            # Execute manipulation policy rollout (following reference: train_utils.py:1096-1106)
            # Get language instruction from env (following reference: train_utils.py:342)
            # If env doesn't have lang, pass empty string (lang encoder will handle it)
            if hasattr(unwrapped_env, '_ep_lang_str'):
                lang = unwrapped_env._ep_lang_str
            elif hasattr(env, '_ep_lang_str'):
                lang = env._ep_lang_str
            else:
                lang = ""  # Empty string as fallback
            rollout_policy.start_episode(lang=lang)
            success = False
            # Get horizon from JSON config (following reference: train_utils.py:119)
            # Priority: train.data[0].horizon > experiment.rollout.horizon
            if hasattr(config.train.data[0], 'get') and 'horizon' in config.train.data[0]:
                horizon = config.train.data[0]['horizon']
            else:
                horizon = config.experiment.rollout.horizon
            
            for step_i in range(horizon):
                # Get action from policy
                ac = rollout_policy(ob=obs_dict, goal=None)
                
                # Execute action
                obs_dict, r, done, info = env.step(ac)
                
                # Render if enabled
                if render_enabled and hasattr(unwrapped_env, 'viewer') and unwrapped_env.viewer is not None:
                    unwrapped_env.render()
                
                # Check success
                if info.get('is_success', {}).get('task', False):
                    success = True
                    break
                
                if done:
                    break
            
            # Only save successful episodes (following reference implementation)
            # Note: Point cloud was already captured BEFORE rollout (step 3-3)
            if success:
                # Save the point cloud and metadata
                # Note: target_pose is the absolute pose (se2_origin + target_se2_delta)
                target_pose_abs = se2_origin + target_se2_delta
                
                # Save point cloud (already captured before rollout)
                collector.save_episode_pointcloud(
                    pcd=pcd,
                    episode_id=successful_episodes,
                    target_pose=target_pose_abs,
                    depth_cameras=depth_cameras,
                    env=env,
                    algo_name=config.algo_name
                )
                
                # Save metadata after each successful episode (for incremental collection)
                collector.save_metadata()
                
                # Update progress
                successful_episodes += 1
                pbar.update(1)
                print(f"Attempt {episode_attempt}: SUCCESS (saved as episode {successful_episodes-1})")
            else:
                print(f"Attempt {episode_attempt}: FAILED (not saved)")
    
    # Final save (redundant but safe)
    collector.save_metadata()
    print(f"\nCollection complete: {successful_episodes} valid episodes collected (out of {episode_attempt} attempts)")
    
    # Close environment
    env.close()
    
    print(f"Data collection complete: {cfg.benchmark.save_dir}")


if __name__ == "__main__":
    main()
