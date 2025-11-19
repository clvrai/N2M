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


def run_rollout_with_predictor(
    env,
    policy: RolloutPolicy,
    predictor: BasePredictor,
    config: Dict[str, Any],
    algo_name: str,
    initial_pose: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Execute one episode rollout with predictor and policy.
    
    Following reference implementation (train_utils.py:836-1110).
    
    Args:
        env: RoboCasa environment (wrapped)
        policy: Manipulation policy (RolloutPolicy)
        predictor: Navigation predictor
        config: Rollout configuration
        algo_name: Algorithm name ('act', 'bc', 'diffusion', etc.)
        initial_pose: Optional initial randomized pose
        
    Returns:
        result: Dictionary with rollout statistics
    """
    start_time = time.time()
    
    # STEP 1: Reset environment (train_utils.py:841)
    ob_dict = env.reset()
    
    # Get unwrapped env (train_utils.py:845-849)
    is_act_policy = algo_name == "act"
    if is_act_policy:
        easy_env = env.env
    else:
        easy_env = env.env.env
    easy_robot = easy_env.robots[0]
    
    # Get SE2 origin (train_utils.py:850)
    se2_origin = obs_to_SE2(ob_dict, algorithm_name=algo_name)
    ac = np.zeros(12)
    
    # Optional: render if enabled
    render_enabled = config.get('render', False)
    if render_enabled:
        easy_env.render()
        env.step(ac)
    
    # STEP 2: Set DETECT mode for robot-mounted camera (train_utils.py:909)
    # Only if predictor needs it (e.g., not for blank predictor)
    if hasattr(predictor, 'needs_detect_mode') and predictor.needs_detect_mode():
        from benchmark.utils.sample_utils import arm_fake_controller
        arm_fake_controller(easy_env, "DETECT")
    
    # STEP 3: Move robot away to capture scene (train_utils.py:862 or 917)
    # Only move robot if predictor needs it (e.g., mobipi for 3DGS)
    # Blank predictor doesn't need this step
    if hasattr(predictor, 'needs_robot_removal') and predictor.needs_robot_removal():
        easy_env.sim.data.qpos[easy_robot._ref_base_joint_pos_indexes] = qpos_command_wrapper(np.array([0.0, -50.0, 0.0]))
        easy_env.sim.forward()
        ob_dict, _, _, _ = env.step(ac)
    
    # STEP 4: Capture point cloud / observation for predictor (train_utils.py:880-887 or 921-927)
    # The predictor will handle this internally based on its needs
    
    # STEP 5: Determine initial navigation pose if randomizing (train_utils.py:930-936)
    prediction_start = time.time()
    
    if initial_pose is not None:
        # Use randomized initial pose from task_area_randomization
        target_se2_relative = initial_pose
    else:
        # Start from origin (0,0,0)
        target_se2_relative = np.array([0.0, 0.0, 0.0])
    
    # Teleport to initial pose
    easy_env.sim.data.qpos[easy_robot._ref_base_joint_pos_indexes] = qpos_command_wrapper(target_se2_relative)
    easy_env.sim.forward()
    ob_dict, _, _, _ = env.step(ac)
    
    # STEP 6: Call predictor to get target pose (train_utils.py:938-1050)
    # Get current SE2 pose after moving to initial pose
    current_se2 = obs_to_SE2(ob_dict, algorithm_name=algo_name)
    
    # Call predictor
    predictor.reset()
    predicted_pose, done, pred_info = predictor.predict(
        observation=ob_dict,
        current_pose=current_se2,
        env_info={'env': easy_env, 'se2_origin': se2_origin}
    )
    
    # Handle pose delta vs absolute pose
    if pred_info.get('is_pose_delta', False):
        final_se2_relative = current_se2 - se2_origin + predicted_pose
    else:
        final_se2_relative = predicted_pose - se2_origin
    
    # Teleport to predicted pose (train_utils.py:1047-1050)
    easy_env.sim.data.qpos[easy_robot._ref_base_joint_pos_indexes] = qpos_command_wrapper(final_se2_relative)
    easy_env.sim.forward()
    ob_dict, _, _, _ = env.step(ac)
    
    prediction_time = time.time() - prediction_start
    
    # STEP 7: Set MANIPULATION mode (train_utils.py:1052)
    if hasattr(predictor, 'needs_detect_mode') and predictor.needs_detect_mode():
        from benchmark.utils.sample_utils import arm_fake_controller
        arm_fake_controller(easy_env, "MANIPULATION")
    
    # STEP 8: Wait for robot to stabilize (train_utils.py:1055-1057)
    for _ in range(5):
        ac = np.zeros(12)
        ob_dict, r, done, info = env.step(ac)
    
    # Get language instruction (train_utils.py:342)
    if hasattr(easy_env, '_ep_lang_str'):
        lang = easy_env._ep_lang_str
    elif hasattr(env, '_ep_lang_str'):
        lang = env._ep_lang_str
    else:
        lang = ""
    policy.start_episode(lang=lang)
    
    # STEP 9: Execute manipulation policy (train_utils.py:1096-1108)
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
            easy_env.render()
        
        # Check success
        if info.get('is_success', {}).get('task', False):
            success = True
            break
        
        if done:
            break
    
    manipulation_time = time.time() - manipulation_start
    total_time = time.time() - start_time
    
    # Get final pose
    final_se2 = obs_to_SE2(ob_dict, algorithm_name=algo_name)
    
    # Build result dictionary
    result = {
        'success': success,
        'num_steps': step_i + 1,
        'initial_pose': (se2_origin + target_se2_relative).tolist(),
        'predicted_pose': (se2_origin + final_se2_relative).tolist(),
        'nav_steps': 1,  # One teleport
        'prediction_time': prediction_time,
        'manipulation_time': manipulation_time,
        'total_time': total_time,
        'predictor_info': pred_info,
    }
    
    return result
