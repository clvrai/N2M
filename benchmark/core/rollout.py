"""Rollout logic for benchmark evaluation.

Migrated from A_ref/N2M-sim/robomimic/robomimic/utils/train_utils.py
Key function: rollout_with_stats_for_SIR → run_rollout_with_predictor

Benchmark evaluation flow (simplified from train_utils.py:840-1110):
1. Reset environment
2. [Optional] Set DETECT mode (only if predictor.needs_detect_mode() == True)
3. [Optional] Move robot away (only if predictor.needs_robot_removal() == True)
   - For benchmark: typically False (predictors use pre-trained models)
   - For data collection: True (need to capture clean scene point cloud)
4. Call predictor to get target pose (predictor handles its own observations)
5. Teleport robot to predicted pose
6. [Optional] Set MANIPULATION mode (only if DETECT mode was used)
7. Execute manipulation policy

Note: Most predictors (blank, n2m, lelan, reachability) don't need robot removal
      during benchmark evaluation, as they use pre-trained models or don't need scene capture.
      Only mobipi (with live 3DGS) needs robot removal.
"""

import time
import numpy as np
from typing import Dict, Any, Optional

from benchmark.utils.transform_utils import qpos_command_wrapper
from benchmark.utils.obs_utils import obs_to_SE2
from benchmark.predictor.base import BasePredictor
from robomimic.algo import RolloutPolicy
from benchmark.utils.navigation_utils import teleport_robot_to_target
from robomimic.envs.wrappers import FrameStackWrapper
from robocasa.environments.kitchen.single_stage.kitchen_doors import OpenSingleDoor
from benchmark.utils.sample_utils import arm_fake_controller
from benchmark.utils.collision_utils import CollisionChecker

def run_rollout_with_predictor(
    env: FrameStackWrapper,
    policy: RolloutPolicy,
    predictor: BasePredictor,
    config: Dict[str, Any],
    collision_checker: CollisionChecker,
    algo_name: str,
    se2_initial: np.ndarray,    # also the origin of the robot coordinate system in global coordinate system
    se2_randomized: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Execute one episode rollout with predictor and policy.
    
    Following reference implementation (train_utils.py:836-1110).
    
    Args:
        env: RoboCasa environment (wrapped)
        policy: Manipulation policy (RolloutPolicy)
        predictor: Navigation predictor
        config: Rollout configuration
        algo_name: Algorithm name ('act', 'bc', 'diffusion', etc.)
        se2_initial: Initial SE2 pose of robot after reset [x, y, theta]
        se2_randomized: Optional randomized SE2 pose for task area randomization [x, y, theta]
        
    Returns:
        result: Dictionary with rollout statistics
    """
    start_time = time.time()
    
    # Get unwrapped env (train_utils.py:845-849)
    unwrapped_env: OpenSingleDoor 

    is_act_policy = algo_name == "act"
    if is_act_policy:
        unwrapped_env = env.env
    else:
        unwrapped_env = env.env.env
    robot = unwrapped_env.robots[0]
    
    ac = np.zeros(12)
    
    # Optional: render if enabled
    render_enabled = config.get('render', False)
    if render_enabled:
        unwrapped_env.render()
        unwrapped_env.viewer.viewer.cam.type = 0
        env.step(ac)

    
    # if hasattr(predictor, 'needs_detect_mode') and predictor.needs_detect_mode():
    #     from benchmark.utils.sample_utils import arm_fake_controller
    #     arm_fake_controller(unwrapped_env, "DETECT")
    
    # if hasattr(predictor, 'needs_robot_removal') and predictor.needs_robot_removal():
    #     teleport_robot_to_target(unwrapped_env, np.array([20.0, 0.0, 0.0]), se2_initial)
    #     ob_dict, _, _, _ = env.step(ac)

    
    # STEP 1: move to sampled pose se2_randomized
    teleport_robot_to_target(unwrapped_env, se2_randomized, se2_initial)
    ob_dict, _, _, _ = env.step(ac)

    # STEP 2: move to sampled pose se2_randomized
    arm_fake_controller(unwrapped_env, "DETECT")
    prediction_start = time.time()
    result_dict = predictor.predict(
        se2_initial =  se2_initial,
        se2_randomized = se2_randomized,
        collision_checker=collision_checker
    )
    prediction_time = time.time() - prediction_start

    # Handle pose delta vs absolute pose
    if result_dict['is_ego']:
        # convert to global coordinates
        se2_predicted = result_dict['se2_predicted']
        global_se2_predicted = np.zeros(3)
        global_se2_predicted[0] = se2_randomized[0] + se2_predicted[0] * np.cos(se2_randomized[2]) - se2_predicted[1] * np.sin(se2_randomized[2])
        global_se2_predicted[1] = se2_randomized[1] + se2_predicted[0] * np.sin(se2_randomized[2]) + se2_predicted[1] * np.cos(se2_randomized[2])
        global_se2_predicted[2] = se2_randomized[2] + se2_predicted[2]
        se2_predicted = global_se2_predicted
    else:
        se2_predicted = result_dict['se2_predicted']
    
    print("[Rollout-summary-start]========================")
    print("Prediction time: {:.2f} s".format(prediction_time))
    print("initial pose:\t", se2_initial)
    print("randomized pose:\t", se2_randomized)
    print("predicted pose:\t", se2_predicted)
    print("[Rollout-summary-end]========================")

    # STEP 3: teleport to predicted region
    teleport_robot_to_target(unwrapped_env, se2_predicted, se2_initial)
    ob_dict, _, _, _ = env.step(ac)
    arm_fake_controller(unwrapped_env, "MANIPULATION")  # switch to manipulation mode.
    for _ in range(10):     # flush the observation buffer
        ac = np.zeros(12)
        ob_dict, _, _, _ = env.step(ac)
    
    # STEP 4: call manipulation policy
    # Get language instruction (train_utils.py:342)
    if hasattr(unwrapped_env, '_ep_lang_str'):
        lang = unwrapped_env._ep_lang_str
    elif hasattr(env, '_ep_lang_str'):
        lang = env._ep_lang_str
    else:
        lang = ""
    policy.start_episode(lang=lang)
    
    manipulation_start = time.time()
    success = False
    # Get horizon from config (passed from JSON config in run_benchmark.py)
    horizon = config.get('horizon', 500)
    
    for step_i in range(horizon):
        # Get action from policy
        ac = policy(ob=ob_dict, goal=None)
        
        # Execute action
        ob_dict, r, done, info = env.step(ac)
        
        # Render if enabled
        if render_enabled:
            unwrapped_env.render()
        
        # Check success
        if info.get('is_success', {}).get('task', False):
            success = True
            break
        
        if done:
            break
    
    manipulation_time = time.time() - manipulation_start
    
    result = {
        'success': success,
        'num_steps': step_i + 1,
        'prediction_time': prediction_time,
        'manipulation_time': manipulation_time,
        'pose_info': {
            "se2_initial": se2_initial.tolist(),
            "se2_randomized": se2_randomized.tolist(),
            "se2_predicted": se2_predicted.tolist(),
        },
        'extra_info': result_dict['extra_info'] if 'extra_info' in result_dict else None
    }
    
    return result
