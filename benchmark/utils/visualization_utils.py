"""Visualization utilities for benchmark.

Basic visualization tools - can be extended later.
"""

import numpy as np
import imageio
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def save_episode_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30
):
    """Save episode frames as video.
    
    Args:
        frames: List of RGB frames (H, W, 3) with values in [0, 255]
        output_path: Output video file path (e.g., 'episode.mp4')
        fps: Frames per second
    """
    imageio.mimsave(output_path, frames, fps=fps)


def visualize_poses_on_topdown(
    predicted_poses: List[np.ndarray],
    initial_pose: Optional[np.ndarray] = None,
    final_pose: Optional[np.ndarray] = None,
    environment_bounds: Optional[tuple] = None,
    output_path: str = 'topdown_poses.png'
):
    """Visualize poses on topdown map.
    
    Args:
        predicted_poses: List of SE2 poses [x, y, theta]
        initial_pose: Initial robot pose
        final_pose: Final robot pose after manipulation
        environment_bounds: (x_min, x_max, y_min, y_max)
        output_path: Output image path
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set bounds
    if environment_bounds is not None:
        x_min, x_max, y_min, y_max = environment_bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    # Plot predicted poses
    for i, pose in enumerate(predicted_poses):
        x, y, theta = pose
        
        # Plot position
        ax.plot(x, y, 'bo', markersize=8, label='Predicted' if i == 0 else '')
        
        # Plot orientation arrow
        dx = 0.2 * np.cos(theta)
        dy = 0.2 * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    # Plot initial pose
    if initial_pose is not None:
        x, y, theta = initial_pose
        ax.plot(x, y, 'go', markersize=10, label='Initial')
        dx = 0.2 * np.cos(theta)
        dy = 0.2 * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='green', ec='green')
    
    # Plot final pose
    if final_pose is not None:
        x, y, theta = final_pose
        ax.plot(x, y, 'ro', markersize=10, label='Final')
        dx = 0.2 * np.cos(theta)
        dy = 0.2 * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Robot Poses (Top-down View)')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_gmm_distribution(
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
    samples: Optional[np.ndarray] = None,
    output_path: str = 'gmm_distribution.png'
):
    """Visualize GMM distribution (for N2M predictor).
    
    Args:
        means: GMM means, shape (num_modes, 2 or 3)
        covariances: GMM covariances, shape (num_modes, 2 or 3, 2 or 3)
        weights: GMM weights, shape (num_modes,)
        samples: Sampled poses, shape (num_samples, 2 or 3)
        output_path: Output image path
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Only visualize x, y (ignore theta)
    means_2d = means[:, :2]
    cov_2d = covariances[:, :2, :2]
    
    # Plot GMM components
    for i, (mean, cov, weight) in enumerate(zip(means_2d, cov_2d, weights)):
        # Plot mean
        ax.plot(mean[0], mean[1], 'rx', markersize=12, markeredgewidth=2)
        
        # Plot covariance ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues)
        
        ellipse = patches.Ellipse(
            mean, width, height, angle=angle,
            fill=False, edgecolor='red', linewidth=2, alpha=weight
        )
        ax.add_patch(ellipse)
    
    # Plot samples
    if samples is not None:
        samples_2d = samples[:, :2]
        ax.plot(samples_2d[:, 0], samples_2d[:, 1], 'b.', markersize=4, alpha=0.5, label='Samples')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('GMM Distribution (N2M Prediction)')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
