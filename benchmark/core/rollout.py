"""Rollout logic for benchmark.

Migrated from A_ref/N2M-sim/robomimic/robomimic/utils/train_utils.py
Key function: rollout_with_stats_for_SIR → run_rollout_with_predictor
"""

import time
import numpy as np
from typing import Dict, Any, Optional

from benchmark.utils.navigation_utils import teleport_robot_to_pose, get_current_robot_pose
from benchmark.predictor.base import BasePredictor
from benchmark.policy.base import BasePolicy


def run_rollout_with_predictor(
    env,
    policy: BasePolicy,
    predictor: BasePredictor,
    config: Dict[str, Any],
    initial_pose: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Execute one episode rollout with predictor and policy.
    
    Unified interface supporting both one-shot and iterative predictors.
    
    Args:
        env: RoboCasa environment
        policy: Manipulation policy
        predictor: Navigation predictor
        config: Rollout configuration with keys:
            - horizon: Maximum steps for manipulation
            - max_nav_steps: Maximum navigation steps (for iterative predictors)
            - randomize_initial_pose: Whether to randomize initial base pose
        initial_pose: Initial base pose (if not randomizing)
        
    Returns:
        result: Dictionary with:
            - success: Whether task succeeded
            - num_steps: Number of manipulation steps
            - initial_pose: Initial robot pose
            - predicted_pose: Final predicted pose
            - nav_steps: Number of navigation steps taken
            - prediction_time: Total time spent on prediction
            - manipulation_time: Time spent on manipulation
            - predictor_info: Additional info from predictor
    """
    start_time = time.time()
    
    # 1. Reset environment
    obs = env.reset()
    
    # 2. Set initial base position
    if initial_pose is not None:
        teleport_robot_to_pose(env, initial_pose)
        obs = env.get_observation()
        initial_pose = initial_pose.copy()
    else:
        initial_pose = get_current_robot_pose(env)
    
    # 3. Predict target pose (unified iterative interface)
    predictor.reset()
    current_pose = initial_pose.copy()
    nav_step = 0
    max_nav_steps = config.get('max_nav_steps', 500)
    
    prediction_start = time.time()
    predictor_info = {}
    
    # Unified iterative loop - works for both one-shot and iterative predictors
    while nav_step < max_nav_steps:
        # Get current observation and pose
        obs = env.get_observation()
        current_pose = get_current_robot_pose(env)
        
        # Get environment info (target object, etc.)
        from benchmark.env.env_utils import get_target_object_info
        env_info = get_target_object_info(env)
        
        # Call predictor
        predicted_pose, done, pred_info = predictor.predict(
            observation=obs,
            current_pose=current_pose,
            env_info=env_info
        )
        
        # Handle pose delta vs absolute pose
        if pred_info.get('is_pose_delta', False):
            # Pose delta - add to current pose
            next_pose = current_pose + predicted_pose
        else:
            # Absolute pose - use directly
            next_pose = predicted_pose
        
        # Teleport to new position
        teleport_robot_to_pose(env, next_pose)
        nav_step += 1
        
        # Store predictor info
        predictor_info.update(pred_info)
        
        # Check if predictor is done
        if done:
            break
    
    # Get final predicted pose
    predicted_pose = get_current_robot_pose(env)
    prediction_time = time.time() - prediction_start
    
    # 4. Execute manipulation policy
    policy.reset()
    obs = env.get_observation()
    
    manipulation_start = time.time()
    success = False
    horizon = config.get('horizon', 500)
    
    for step in range(horizon):
        # Predict action
        action = policy.predict_action(obs)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Check success
        if info.get('success', False):
            success = True
            break
        
        if done:
            break
    
    manipulation_time = time.time() - manipulation_start
    total_time = time.time() - start_time
    
    # 5. Build result dictionary
    result = {
        'success': success,
        'num_steps': step + 1,
        'initial_pose': initial_pose.tolist(),
        'predicted_pose': predicted_pose.tolist(),
        'nav_steps': nav_step,
        'prediction_time': prediction_time,
        'manipulation_time': manipulation_time,
        'total_time': total_time,
        'predictor_info': predictor_info,
    }
    
    return result
