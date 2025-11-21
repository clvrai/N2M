"""Sampling utilities for benchmark.

Migrated from A_ref/N2M-sim/robomimic/robomimic/utils/sample_utils.py
"""

import numpy as np
from typing import Tuple, Optional, Dict
from benchmark.utils.collision_utils import CollisionChecker


def sample_collision_free_pose(
    collision_checker: CollisionChecker,
    pose_range: Dict[str, Tuple[float, float]],
    se2_initial: np.ndarray = np.array([0.0, 0.0, 0.0]),
    max_tries: int = 100,
    visualize: bool = False,
    save_path: Optional[str] = None,
    object_pos: Optional[np.ndarray] = None,
    check_visibility: bool = False,
    check_boundary: bool = True
) -> Optional[np.ndarray]:
    """Sample a collision-free SE2 pose with optional visibility and boundary checks.
    
    Following reference: sample_utils.py get_random_target_se2_with_visibility_check (line 680-699)
    
    Args:
        collision_checker: CollisionChecker instance
        pose_range: Dictionary with 'x', 'y', 'theta' ranges
            Example: {'x': [-0.5, 0.5], 'y': [-0.5, 0.5], 'theta': [-np.pi, np.pi]}
        se2_initial: Origin pose to add to sampled delta
        max_tries: Maximum number of sampling attempts
        visualize: Whether to visualize the sampled pose
        save_path: Optional path to save visualization (e.g., 'debug/collision_check.png')
        object_pos: Object position [x, y] or [x, y, z] for visibility check
        check_visibility: Whether to check if object is visible from camera
        check_boundary: Whether to check if pose is within point cloud boundaries
        
    Returns:
        Collision-free pose, or None if failed after max_tries
    """
    for attempt in range(max_tries):
        # Sample random delta
        dx = np.random.uniform(*pose_range['x'])
        dy = np.random.uniform(*pose_range['y'])
        dtheta = np.random.uniform(*pose_range['theta'])
        
        # Compute absolute pose
        pose = se2_initial + np.array([dx, dy, dtheta])
        
        # Check collision (following reference implementation semantics)
        # check_collision returns True if NO collision (pose is valid)
        collision_free = collision_checker.check_collision(pose)
        if not collision_free:
            continue
        
        # Check visibility if requested and object position provided (following reference line 690)
        if check_visibility and object_pos is not None:
            visible = collision_checker.check_object_visibility(pose, object_pos)
            if not visible:
                continue
        
        # Check boundary if requested (following reference line 690)
        if check_boundary:
            x_range = (collision_checker.pcd_min_value[0], collision_checker.pcd_max_value[0])
            y_range = (collision_checker.pcd_min_value[1], collision_checker.pcd_max_value[1])
            within_boundary = collision_checker.check_boundary(pose, x_range, y_range)
            if not within_boundary:
                continue
        
        # All checks passed!
        # Visualize if requested (only save, don't show to avoid blocking)
        if visualize and save_path is not None:
            collision_checker.visualize_occupancy_and_rectangle(
                se2_initial=se2_initial,
                sampled_pose=pose,
                pose_range=pose_range,
                save_path=save_path,
                show_plot=False  # Don't show plot to avoid blocking
            )
        
        # Return delta (relative pose), not absolute pose
        # Following reference: get_random_target_se2_with_reachability_check returns se2_delta
        # delta_pose = pose - se2_initial
        return pose
    
    # Failed to sample collision-free pose
    print(f"Warning: Failed to sample collision-free pose after {max_tries} attempts")
    return None


def sample_from_gmm(
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
    num_samples: int,
    collision_checker: Optional[CollisionChecker] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample poses from Gaussian Mixture Model.
    
    Used for N2M predictor.
    
    Args:
        means: GMM means, shape (num_modes, 3)
        covariances: GMM covariances, shape (num_modes, 3, 3)
        weights: GMM weights, shape (num_modes,)
        num_samples: Number of samples to draw
        collision_checker: Optional collision checker to filter samples
        
    Returns:
        samples: Sampled poses, shape (num_valid_samples, 3)
        scores: Log probability scores for each sample
    """
    num_modes = means.shape[0]
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Sample mode indices according to weights
    mode_indices = np.random.choice(num_modes, size=num_samples, p=weights)
    
    # Sample from each selected mode
    samples = []
    scores = []
    
    for mode_idx in mode_indices:
        # Sample from multivariate Gaussian
        sample = np.random.multivariate_normal(
            means[mode_idx], 
            covariances[mode_idx]
        )
        
        # Check collision if checker provided
        if collision_checker is not None:
            if collision_checker.check_collision(sample):
                continue  # Skip colliding samples
        
        samples.append(sample)
        
        # Compute log probability (simplified, just for this mode)
        diff = sample - means[mode_idx]
        inv_cov = np.linalg.inv(covariances[mode_idx])
        log_prob = -0.5 * diff @ inv_cov @ diff
        scores.append(log_prob)
    
    if len(samples) == 0:
        return np.array([]), np.array([])
    
    return np.array(samples), np.array(scores)


def select_best_sample(
    samples: np.ndarray,
    scores: np.ndarray,
    selection_mode: str = 'max_score'
) -> int:
    """Select best sample from candidates.
    
    Args:
        samples: Candidate samples, shape (num_samples, 3)
        scores: Scores for each sample
        selection_mode: Selection criterion ('max_score', 'random')
        
    Returns:
        Index of selected sample
    """
    if selection_mode == 'max_score':
        return np.argmax(scores)
    elif selection_mode == 'random':
        return np.random.choice(len(samples))
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")
