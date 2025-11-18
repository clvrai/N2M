"""Environment utilities for benchmark.

Migrated from A_ref/N2M-sim/scripts/1_data_collection_with_rollout.py
"""

import numpy as np
from typing import Dict, List, Any, Optional
import robomimic.utils.env_utils as RobomimicEnvUtils
import robomimic.utils.file_utils as FileUtils


def create_env_from_config(
    env_config: Dict[str, Any],
    seed: int,
    render: bool = False,
    render_offscreen: bool = True
):
    """Create RoboCasa environment from config.
    
    Following A_ref/N2M-sim/scripts/1_data_collection_with_rollout.py structure:
    - Load JSON config
    - Build env_meta with env_meta_update_dict from JSON
    - Create environment using robomimic utils
    
    Args:
        env_config: Environment configuration dict (from Hydra)
        seed: Random seed
        render: Whether to render on-screen
        render_offscreen: Whether to render offscreen (for RGB observations)
        
    Returns:
        env: Created RoboCasa environment
    """
    import json
    import os
    from robomimic.utils.script_utils import deep_update
    
    # Load JSON config (following reference implementation)
    json_config_path = os.path.expanduser(env_config['json_config_path'])
    with open(json_config_path, 'r') as f:
        json_config = json.load(f)
    
    # Load env_meta from dataset (CRITICAL: following reference implementation exactly)
    # Reference: A_ref/N2M-sim/scripts/1_data_collection_with_rollout.py lines 76-93
    env_meta = None
    
    if 'train' in json_config and 'data' in json_config['train']:
        dataset_cfg = json_config['train']['data'][0] if json_config['train']['data'] else None
        if dataset_cfg:
            dataset_path = os.path.expanduser(dataset_cfg.get('path', ''))
            if os.path.exists(dataset_path):
                # Load complete env_meta from dataset hdf5 (includes all critical parameters)
                import robomimic.utils.file_utils as FileUtils
                env_meta = FileUtils.get_env_metadata_from_dataset(
                    dataset_path=dataset_path,
                    ds_format=json_config.get('train', {}).get('data_format', 'robomimic')
                )
                
                # Populate language instruction for env in env_meta (reference line 87)
                env_meta["env_lang"] = dataset_cfg.get("lang", None)
                
                # Apply env_meta updates (reference lines 90-92)
                # IMPORTANT: These updates should AUGMENT, not REPLACE the env_meta
                from robomimic.utils.script_utils import deep_update
                deep_update(env_meta, dataset_cfg.get("env_meta_update_dict", {}))
                if 'experiment' in json_config and 'env_meta_update_dict' in json_config['experiment']:
                    deep_update(env_meta, json_config['experiment']['env_meta_update_dict'])
    
    # Fallback: if no dataset found, build minimal env_meta (for data collection without existing dataset)
    if env_meta is None:
        env_meta = {
            "env_name": env_config['name'],
            "type": 1,  # EnvType.ROBOSUITE_TYPE
            "env_kwargs": {
                "robots": "PandaMobile",  # Default for RoboCasa
            }
        }
        # Apply env_meta updates from JSON config
        from robomimic.utils.script_utils import deep_update
    if 'experiment' in json_config and 'env_meta_update_dict' in json_config['experiment']:
        print(f"[DEBUG] Before deep_update: has_renderer={env_meta.get('env_kwargs', {}).get('has_renderer')}")
        print(f"[DEBUG] Applying env_meta_update_dict: {json_config['experiment']['env_meta_update_dict']}")
        deep_update(env_meta, json_config['experiment']['env_meta_update_dict'])
        print(f"[DEBUG] After deep_update: has_renderer={env_meta.get('env_kwargs', {}).get('has_renderer')}")
    
    # Override with YAML config (YAML takes precedence over JSON)
    if 'layout_and_style_ids' in env_config:
        print(f"[DEBUG] Overriding layout_and_style_ids from YAML: {env_config['layout_and_style_ids']}")
        if 'env_kwargs' not in env_meta:
            env_meta['env_kwargs'] = {}
        env_meta['env_kwargs']['layout_and_style_ids'] = env_config['layout_and_style_ids']
    
    # Override has_renderer based on YAML render setting (YAML takes precedence)
    if 'render' in env_config:
        yaml_render = env_config['render']
        print(f"[DEBUG] Overriding has_renderer from YAML render setting: {yaml_render}")
        if 'env_kwargs' not in env_meta:
            env_meta['env_kwargs'] = {}
        env_meta['env_kwargs']['has_renderer'] = yaml_render
    
    # Get observation config from JSON
    obs_modalities = json_config.get('observation', {}).get('modalities', {}).get('obs', {})
    all_obs_keys = []
    for modality_keys in obs_modalities.values():
        if isinstance(modality_keys, list):
            all_obs_keys.extend(modality_keys)
    
    # Check if using images
    use_image_obs = len(obs_modalities.get('rgb', [])) > 0
    
    # Get render settings with priority: function params > YAML config > JSON config
    # Priority order (highest to lowest):
    # 1. Function parameters (if explicitly provided)
    # 2. YAML config (env_config['render'])
    # 3. JSON config (json_config['experiment']['render'])
    
    json_render = json_config.get('experiment', {}).get('render', False)
    json_render_offscreen = json_config.get('experiment', {}).get('render_video', False)
    
    # Get YAML config (if available)
    yaml_render = env_config.get('render', None)
    
    # Determine final render settings
    # Function parameter takes precedence, then YAML, then JSON
    if render is not None:
        final_render = render
        print(f"[DEBUG] Using render from function parameter: {final_render}")
    elif yaml_render is not None:
        final_render = yaml_render
        print(f"[DEBUG] Using render from YAML config: {final_render}")
    else:
        final_render = json_render
        print(f"[DEBUG] Using render from JSON config: {final_render}")
    
    final_render_offscreen = render_offscreen if render_offscreen is not None else json_render_offscreen
    # Note: offscreen renderer will be auto-enabled by EnvRobosuite if use_image_obs=True
    
    # Print detailed configuration for debugging
    print("\n" + "="*80)
    print("ENVIRONMENT CREATION PARAMETERS (N2M-benchmark)")
    print("="*80)
    print(f"env_name: {env_config['name']}")
    print(f"render: {final_render}")
    print(f"render_offscreen: {final_render_offscreen}")
    print(f"use_image_obs: {use_image_obs}")
    print(f"seed: {seed}")
    print(f"\nenv_meta keys: {list(env_meta.keys())}")
    print(f"env_meta['type']: {env_meta.get('type')}")
    print(f"env_meta['env_name']: {env_meta.get('env_name')}")
    
    if 'env_kwargs' in env_meta:
        print(f"\nenv_meta['env_kwargs'] keys: {list(env_meta['env_kwargs'].keys())}")
        for key in ['robots', 'has_renderer', 'has_offscreen_renderer', 'use_camera_obs', 
                    'use_object_obs', 'camera_names', 'camera_heights', 'camera_widths', 
                    'camera_depths', 'render', 'render_camera', 'control_freq']:
            if key in env_meta['env_kwargs']:
                print(f"  {key}: {env_meta['env_kwargs'][key]}")
    print("="*80 + "\n")
    
    # Create environment using robomimic utils (same as reference)
    env = RobomimicEnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_config['name'],
        render=final_render,
        render_offscreen=final_render_offscreen,
        use_image_obs=use_image_obs,
        seed=seed,
    )
    
    # Apply environment wrappers (CRITICAL: adds FrameStackWrapper for BC policies)
    # Following reference implementation: A_ref/N2M-sim/scripts/1_data_collection_with_rollout.py:136
    from robomimic.config import config_factory
    config = config_factory(json_config["algo_name"])
    with config.values_unlocked():
        config.update(json_config)
    env = RobomimicEnvUtils.wrap_env_from_config(env, config=config)
    
    return env


def get_target_object_info(env) -> Dict[str, Any]:
    """Get target object information from environment.
    
    Args:
        env: RoboCasa environment
        
    Returns:
        info: Dictionary with target object information
            - 'position': 3D position of target object
            - 'name': Name of target fixture/object
    """
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env.env
    
    # Get target fixture name (usually stored in init_robot_base_pos)
    if hasattr(unwrapped_env, 'init_robot_base_pos'):
        fixture_name = unwrapped_env.init_robot_base_pos.name
    else:
        # Fallback: try to get from task
        fixture_name = getattr(unwrapped_env, 'target_fixture_name', None)
    
    # Get fixture position
    if fixture_name is not None and hasattr(unwrapped_env, 'fixtures'):
        fixture_pos = unwrapped_env.fixtures[fixture_name].pos
    else:
        # Fallback: return origin
        fixture_pos = np.array([0.0, 0.0, 1.0])
    
    return {
        'position': fixture_pos,
        'name': fixture_name
    }


def get_env_observation_with_depth(env, camera_names: List[str]) -> Dict[str, np.ndarray]:
    """Get environment observation including depth images.
    
    Args:
        env: RoboCasa environment
        camera_names: List of camera names to capture depth from
        
    Returns:
        observation: Observation dict with depth added
    """
    # Get standard observation
    obs = env.get_observation()
    
    # Add depth for specified cameras
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env.env
    sim = unwrapped_env.sim
    
    for cam_name in camera_names:
        if cam_name in unwrapped_env.camera_names:
            depth_key = f"{cam_name}_depth"
            if depth_key not in obs:
                # Render depth directly from simulator
                cam_idx = unwrapped_env.camera_names.index(cam_name)
                camera_height = unwrapped_env.camera_heights[cam_idx]
                camera_width = unwrapped_env.camera_widths[cam_idx]
                
                # Render depth image using mujoco
                depth = sim.render(
                    camera_name=cam_name,
                    width=camera_width,
                    height=camera_height,
                    depth=True
                )[1]  # render returns (rgb, depth)
                
                obs[depth_key] = depth
    
    return obs
