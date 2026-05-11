"""Mobipi predictor - uses 3D Gaussian Splatting and Bayesian Optimization.

Reference: N2M-benchmark/predictor/mobipi/mobipi
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional, Any
import torch
import h5py
from tqdm import tqdm
from PIL import Image

from benchmark.predictor.base import BasePredictor
from benchmark.utils.collision_utils import CollisionChecker
import os
from glob import glob
from mobipi.utils.io_utils import camel_to_snake_case
from mobipi.scene_model.scene_model import BatchSceneModel
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
from mobipi.utils.env_utils import compute_relative_cam_pose, compute_camera_extrinsics
from mobipi.utils.encoder_utils import DinoEncoder
from mobipi.utils.opt_utils import optimize_pose_batch, normalize as normalize_in_bounds
# Note: Original mobipi uses random uniform sampling, not collision-free sampling

class MobipiPredictor(BasePredictor):
    """Mobipi predictor using 3DGS scene model and optimization.
    
    One-shot predictor that returns done=True on first call.
    """
    
    def __init__(self, hydra_cfg, json_config, env, unwrapped_env):
        """Initialize Mobipi predictor.
        
        Args:
            hydra_cfg: Hydra config
            json_config: Robomimic/Robocasa config
            env: Environment instance (step)
            unwrapped_env: Unwrapped environment instance (forward)
        """
        super().__init__()  # BasePredictor.__init__() takes no arguments
        
        self.hydra_cfg = hydra_cfg
        self.json_config = json_config
        self.env = env
        self.unwrapped_env = unwrapped_env

        self.camera_name = hydra_cfg.predictor.camera_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = hydra_cfg.predictor.image_size
        self.encoder_type = hydra_cfg.predictor.encoder_type
        
        # Will be initialized in load_checkpoint()
        self.batch_scene_model = None
        self.camera_names = None
        self.camera_intrinsics_dict = None
        self.rel_cam_positions = None  # Relative camera positions
        self.rel_cam_mats = None  # Relative camera orientations
        
        # Feature encoder and buffer (initialized in load_policy_dataset)
        self.encoder = None
        self.feature_buffer = None
        
        # Robot images and masks (for compositing with rendered images)
        self.robot_imgs_torch = None
        self.robot_masks_torch = None
        
        self.load_checkpoint()
        self.load_policy_dataset()
        self._get_robot_images_and_masks()
        
    
    def predict(self, se2_initial, se2_randomized, collision_checker: CollisionChecker, episode_id=None):
        """Predict target pose using Mobipi.
        
        Args:
            se2_initial: Initial SE2 pose of robot after reset [x, y, theta]
            se2_randomized: Randomized SE2 pose for task area randomization [x, y, theta]
            collision_checker: Collision checker instance
            
        Returns:
            result: Dict with prediction metadata
        """
        print(f"\n============= Running Bayesian Optimization =============")
        # Following original mobipi: eval_mobipi.py Line 427
        # Original uses 2500 samples with uniform random sampling (no collision check)
        num_init_samples = self.hydra_cfg.predictor.num_init_samples
        # Following original mobipi: eval_mobipi.py Line 88, 445
        # Original uses 500 BO samples = 100 iterations with batch_size=5
        bo_num_samples = self.hydra_cfg.predictor.bo_num_samples


        # Initialize timing variables (will be accumulated in score_function)
        rendering_time = 0.0
        feature_extraction_time = 0.0
        collision_check_time = 0.0
        start_time = time.time()
        
        # Track best pose for real-time visualization
        best_score_so_far = -float('inf')
        best_pose_so_far = None
        best_rendered_images_so_far = None
        best_final_images_so_far = None
        debug_dataset_dir = "/home/kaixin/workbench/N2M-benchmark/debug_dataset"
        os.makedirs(debug_dataset_dir, exist_ok=True)
        
        # Get search bounds from config
        task_area_rand = self.hydra_cfg.benchmark.task_area_randomization
        bounds = [
            (se2_initial[0] + task_area_rand.x[0], se2_initial[0] + task_area_rand.x[1]),
            (se2_initial[1] + task_area_rand.y[0], se2_initial[1] + task_area_rand.y[1]),
            (se2_initial[2] + task_area_rand.theta[0], se2_initial[2] + task_area_rand.theta[1]),
        ]
        print(f"[predictor] Search bounds: {bounds}")
        
        # Generate initial samples using random uniform sampling

        print(f"[predictor] Generating {num_init_samples} random initial samples...")
        
        # Random uniform sampling in normalized space [0, 1]^3
        # Collision checking is handled in the score function (negative score for collisions)
        rng = np.random.RandomState(seed=0)
        initial_samples = rng.uniform(0, 1, size=(num_init_samples, 3))
        print(f"[predictor] Generated {num_init_samples} samples (no collision pre-filtering)")
        
        # Define score function
        def score_function(robot_poses):
            """Score function for Bayesian Optimization.
            
            Args:
                robot_poses: Array of SE2 poses to evaluate, shape (N, 3)
                
            Returns:
                scores: List of scores for each pose
                score_info: Dict with additional information
            """
            nonlocal rendering_time, feature_extraction_time, collision_check_time
            nonlocal best_score_so_far, best_pose_so_far, best_rendered_images_so_far, best_final_images_so_far
            
            num_poses = len(robot_poses)
            scores = []
            
            # Add progress indicator for batches
            # Always show progress for better visibility
            show_progress = num_poses > 5
            iterator = tqdm(robot_poses, desc=f"Scoring {num_poses} poses", leave=False) if show_progress else robot_poses
            
            num_collisions = 0
            # Save debug images for large batches (likely initial samples)
            # Use >= to include exactly 100 samples
            save_debug_images = False  # Save for batches with 50+ samples
            debug_save_count = 0
            max_debug_saves = 10  # Save first 10 valid renders
            
            for idx, pose in enumerate(iterator):
                # Check collision first (fast check before expensive rendering)
                collision_start = time.time()
                is_collision = collision_checker.check_collision(pose)
                collision_check_time += time.time() - collision_start
                
                if is_collision:
                    scores.append(-1.0)  # Collision positions get negative score
                    num_collisions += 1
                    continue
                
                # Render images from this pose
                render_start = time.time()
                robot_pose_torch = torch.tensor(pose, device=self.device, dtype=torch.float32)
                
                # Compute camera extrinsics for each camera
                batch_view_extrinsics = []
                for rel_cam_position, rel_cam_mat in zip(self.rel_cam_positions, self.rel_cam_mats):
                    extrinsics = compute_camera_extrinsics(
                        robot_pose_torch, rel_cam_position, rel_cam_mat
                    )
                    batch_view_extrinsics.append(extrinsics)
                
                # Render from 3DGS
                with torch.no_grad():
                    rendered_images = self.batch_scene_model.render(
                        batch_view_extrinsics, image_size=self.image_size
                    )  # Shape: (num_cameras, H, W, 3)
                
                rendering_time += time.time() - render_start
                
                # Composite robot onto rendered images
                # Formula: final = rendered * (1 - mask) + robot * mask
                final_images = (
                    rendered_images * (1 - self.robot_masks_torch) +
                    self.robot_imgs_torch * self.robot_masks_torch
                )
                
                # Save debug images for first few valid renders
                if save_debug_images and debug_save_count < max_debug_saves:
                    debug_dir = "debug_renders"
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    # Save pose information
                    pose_info_path = f"{debug_dir}/pose_info.txt"
                    with open(pose_info_path, "a") as f:
                        f.write(f"Pose {idx}: x={pose[0]:.3f}, y={pose[1]:.3f}, theta={pose[2]:.3f}\n")
                    
                    # Save rendered images (before and after mask)
                    for cam_idx in range(len(rendered_images)):
                        # Save pure 3DGS rendering (without robot)
                        img_no_robot = (rendered_images[cam_idx].cpu().numpy() * 255).astype(np.uint8)
                        img_pil_no_robot = Image.fromarray(img_no_robot)
                        save_path_no_robot = f"{debug_dir}/render_pose{idx}_cam{cam_idx}_no_robot.png"
                        img_pil_no_robot.save(save_path_no_robot)
                        
                        # Save final composited image (with robot)
                        img_with_robot = (final_images[cam_idx].cpu().numpy() * 255).astype(np.uint8)
                        img_pil_with_robot = Image.fromarray(img_with_robot)
                        save_path_with_robot = f"{debug_dir}/render_pose{idx}_cam{cam_idx}_with_robot.png"
                        img_pil_with_robot.save(save_path_with_robot)
                        
                        # Save robot mask for visualization
                        mask_vis = (self.robot_masks_torch[cam_idx].cpu().numpy() * 255).astype(np.uint8)
                        mask_pil = Image.fromarray(mask_vis.squeeze())
                        save_path_mask = f"{debug_dir}/render_pose{idx}_cam{cam_idx}_mask.png"
                        mask_pil.save(save_path_mask)
                    
                    debug_save_count += 1
                    if debug_save_count == 1:
                        print(f"\n[DEBUG] Saving rendered images to {debug_dir}/")
                        print(f"[DEBUG] Saving 3 versions: no_robot, with_robot, mask")
                        print(f"[DEBUG] Robot mask compositing is now ENABLED!")
                
                # Extract features from final composited images
                feature_start = time.time()
                # Convert to (num_cameras, 3, H, W) - already normalized to [0, 1]
                final_images_tensor = final_images.permute(0, 3, 1, 2)  # (N, 3, H, W)
                
                with torch.no_grad():
                    features = self.encoder(final_images_tensor)  # (num_cameras, feature_dim)
                
                # Average features across cameras
                avg_feature = features.mean(dim=0).cpu().numpy()  # (feature_dim,)
                feature_extraction_time += time.time() - feature_start
                
                # Compute similarity with feature buffer
                score = self._compute_similarity_score(avg_feature)
                scores.append(score)
                
                # Update best pose if this is better
                if score > best_score_so_far:
                    best_score_so_far = score
                    best_pose_so_far = pose.copy()
                    best_rendered_images_so_far = rendered_images.clone()
                    best_final_images_so_far = final_images.clone()
                    
                    # Save current best pose images immediately
                    print(f"\n[DEBUG] New best score: {score:.4f} at pose [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
                    for cam_idx in range(len(best_final_images_so_far)):
                        # Save 3DGS rendering (without robot)
                        img_no_robot = (best_rendered_images_so_far[cam_idx].cpu().numpy() * 255).astype(np.uint8)
                        img_pil_no_robot = Image.fromarray(img_no_robot)
                        img_pil_no_robot.save(f"{debug_dataset_dir}/current_best_cam{cam_idx}_no_robot.png")
                        
                        # Save final composited image (with robot)
                        img_with_robot = (best_final_images_so_far[cam_idx].cpu().numpy() * 255).astype(np.uint8)
                        img_pil_with_robot = Image.fromarray(img_with_robot)
                        img_pil_with_robot.save(f"{debug_dataset_dir}/current_best_cam{cam_idx}_with_robot.png")
                    
                    # Save best pose info
                    with open(f"{debug_dataset_dir}/current_best_info.txt", "w") as f:
                        f.write(f"Current Best Pose: x={pose[0]:.3f}, y={pose[1]:.3f}, theta={pose[2]:.3f}\n")
                        f.write(f"Current Best Score: {score:.4f}\n")
            
            # Print statistics for large batches
            if num_poses > 100:
                print(f"[predictor] Scored {num_poses} poses: {num_collisions} collisions, {num_poses - num_collisions} valid")
            
            score_info = {}
            return scores, score_info
        
        # Run Bayesian Optimization
        # Following original mobipi: eval_mobipi.py Line 445, 452
        # Original uses batch_size=5 with base_estimator="ET" (Extra Trees)
        n_iterations = bo_num_samples // 5  # 100 iterations
        batch_size = 5
        
        print(f"[predictor] Running BO with {n_iterations} iterations, batch_size={batch_size}...")
        print(f"[predictor] Evaluating {num_init_samples} initial samples first (this may take 15-20 minutes)...")
        print(f"[predictor] Total candidates to evaluate: {num_init_samples + bo_num_samples} = {num_init_samples + bo_num_samples}")
        
        # Setup log file path
        log_file_path = f"{self.hydra_cfg.benchmark.results_dir}/mobipi_{self.hydra_cfg.env.name}_initSamples_{num_init_samples}_boSamples_{bo_num_samples}_scene_{self.unwrapped_env.layout_id}_style_{self.unwrapped_env.style_id}/optimize_log_episodeID_{episode_id}.txt"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        print(f"[predictor] Detailed timing log will be saved to: {log_file_path}")
        
        optimize_start = time.time()
        best_pose, history = optimize_pose_batch(
            score_function,
            bounds,
            algorithm="bayesian",
            n_iterations=n_iterations,
            batch_size=batch_size,
            normalize_data=True,
            track_info=True,
            initial_samples=initial_samples,
            base_estimator="ET",  # Following original mobipi: eval_mobipi.py Line 97, 444
            acq_func="LCB",
            acq_func_kwargs=dict(kappa=1.96),
            seed=0,
            log_file=log_file_path,
        )
        optimize_pose_batch_time = time.time() - optimize_start
        
        total_time = time.time() - start_time
        
        # Get best score
        best_score = max(history['sampled_scores'])
        num_candidates = len(history['sampled_scores'])
        
        print(f"[predictor] Best pose: {best_pose.round(3)}")
        print(f"[predictor] Best score: {best_score:.4f}")
        print(f"[predictor] Total candidates evaluated: {num_candidates}")
        print(f"[DEBUG] Best pose images saved in real-time to /home/kaixin/workbench/N2M-benchmark/debug_dataset/current_best_*")
        
        # Calculate overhead time (GP training, BO ask/tell, etc.)
        score_function_time = rendering_time + feature_extraction_time + collision_check_time
        bo_overhead_time = optimize_pose_batch_time - score_function_time
        
        print(f"\n[Timing Summary]")
        print(f"  Total predict time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"  optimize_pose_batch time: {optimize_pose_batch_time:.2f}s ({optimize_pose_batch_time/60:.2f} min)")
        print(f"    ├─ Score function time: {score_function_time:.2f}s ({score_function_time/60:.2f} min)")
        print(f"    │  ├─ Rendering: {rendering_time:.2f}s ({rendering_time/score_function_time*100:.1f}%)")
        print(f"    │  ├─ Feature extraction: {feature_extraction_time:.2f}s ({feature_extraction_time/score_function_time*100:.1f}%)")
        print(f"    │  └─ Collision check: {collision_check_time:.2f}s ({collision_check_time/score_function_time*100:.1f}%)")
        print(f"    └─ BO overhead (GP/Ask/Tell): {bo_overhead_time:.2f}s ({bo_overhead_time/60:.2f} min)")
        print(f"============= Optimization Complete =============")
        
        # Append timing summary to log file
        with open(log_file_path, 'a') as f:
            f.write("\n[PREDICT() TIMING SUMMARY]\n")
            f.write(f"  Total predict time: {total_time:.2f}s ({total_time/60:.2f} min)\n")
            f.write(f"  optimize_pose_batch time: {optimize_pose_batch_time:.2f}s ({optimize_pose_batch_time/60:.2f} min)\n")
            f.write(f"    ├─ Score function time: {score_function_time:.2f}s ({score_function_time/60:.2f} min)\n")
            f.write(f"    │  ├─ Rendering: {rendering_time:.2f}s ({rendering_time/score_function_time*100:.1f}%)\n")
            f.write(f"    │  ├─ Feature extraction: {feature_extraction_time:.2f}s ({feature_extraction_time/score_function_time*100:.1f}%)\n")
            f.write(f"    │  └─ Collision check: {collision_check_time:.2f}s ({collision_check_time/score_function_time*100:.1f}%)\n")
            f.write(f"    └─ BO overhead (GP/Ask/Tell): {bo_overhead_time:.2f}s ({bo_overhead_time/60:.2f} min)\n")
            f.write("\n" + "="*80 + "\n\n")
            f.flush()
        
        # Return in ego frame (best_pose is already in world frame, need to convert to ego)
        # For now, return as world frame and set is_ego=False
        # TODO: Convert to ego frame if needed
        se2_predicted = best_pose
        
        extra_info = {
            # Total times
            'total_predict_time': total_time,
            'optimize_pose_batch_time': optimize_pose_batch_time,
            
            # Score function breakdown (accumulated across all calls)
            'rendering_time': rendering_time,
            'feature_extraction_time': feature_extraction_time,
            'collision_check_time': collision_check_time,
            'score_function_time': score_function_time,
            
            # BO overhead (GP training, ask/tell, etc.)
            'bo_overhead_time': bo_overhead_time,
            
            # Other info
            'best_score': float(best_score),
            'num_candidates_evaluated': num_candidates,
            'optimize_pose_batch_log': log_file_path
        }
        
        result = {
            'is_ego': False,  # Returning world frame coordinates
            'se2_predicted': se2_predicted,
            'extra_info': extra_info
        }
        return result
    
    def _compute_similarity_score(self, query_feature: np.ndarray) -> float:
        """Compute similarity score between query feature and feature buffer.
        
        Args:
            query_feature: Query feature vector (feature_dim,)
            
        Returns:
            score: Similarity score (higher is better)
        """
        # Normalize query feature
        query_norm = query_feature / (np.linalg.norm(query_feature) + 1e-8)
        
        # Normalize buffer features
        buffer_norms = self.feature_buffer / (np.linalg.norm(self.feature_buffer, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarities
        similarities = np.dot(buffer_norms, query_norm)  # (num_demos,)
        
        # Return max similarity (or mean of top-k)
        k = min(5, len(similarities))
        top_k_sim = np.partition(similarities, -k)[-k:]
        score = np.mean(top_k_sim)
        
        return float(score)

    def reset(self):
        """Reset predictor state."""
        pass

    def load_policy_dataset(self):
        """Load policy dataset and build feature buffer.
        
        Note: Original Mobipi flips images from sim.render() using [::-1].
        Dataset images are already flipped during collection (collect_images.py:167).
        So we load them as-is without additional flipping.
        """
        
        print(f"\n============= Loading Training Data and Building Feature Buffer =============")
        
        # Get dataset path from config
        # policy_dataset_path = self.hydra_cfg.predictor.policy_dataset_path    # previously, will choose robomimic
        policy_dataset_path = self.hydra_cfg.policy.dataset_path  # currently, will choose 
        print(f"[predictor] Dataset path: {policy_dataset_path}")
        
        # Get filter key from json config
        filter_key = self.json_config.train.data[0].get("filter_key", None)
        print(f"[predictor] Filter key: {filter_key}")
        
        # Load DINO encoder
        print(f"[predictor] Loading {self.encoder_type} encoder...")
        if self.encoder_type == "dino":
            self.encoder = DinoEncoder(device=self.device, model_name="vit_base_patch16_224_dino")
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")
        
        # Extract initial frame images from all demos
        print(f"[predictor] Extracting initial frame images from HDF5...")
        initial_images = []
        
        with h5py.File(policy_dataset_path, "r") as df:
            # Get demo keys based on filter
            if filter_key is not None:
                demo_keys = list(df["mask"][filter_key])
            else:
                demo_keys = list(df["data"].keys())
            
            print(f"[predictor] Found {len(demo_keys)} demos")
            
            # Extract initial frame for each camera
            for demo_key in tqdm(demo_keys, desc="Loading initial images"):
                for camera_name in self.camera_names:
                    obs_key = f"{camera_name}_image"
                    if obs_key in df["data"][demo_key]["obs"]:
                        # Get first frame (index 0)
                        img = df["data"][demo_key]["obs"][obs_key][0]
                        initial_images.append(img)
        
        initial_images = np.array(initial_images)
        print(f"[predictor] Loaded {len(initial_images)} initial images")
        
        # DEBUG: Save first 10 images to check if they need flipping
        debug_dataset_dir = "/home/kaixin/workbench/N2M-benchmark/debug_dataset"
        os.makedirs(debug_dataset_dir, exist_ok=True)
        print(f"[DEBUG] Saving first 10 dataset images to {debug_dataset_dir}/")
        for i in range(min(10, len(initial_images))):
            img = initial_images[i]
            img_pil = Image.fromarray(img)
            img_pil.save(f"{debug_dataset_dir}/dataset_img_{i:03d}.png")
        print(f"[DEBUG] Saved {min(10, len(initial_images))} images")
        
        # Extract features in batches
        print(f"[predictor] Extracting features...")
        batch_size = 256
        all_features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(initial_images), batch_size), desc="Extracting features"):
                batch = initial_images[i:i+batch_size]
                # Convert to torch tensor: (B, H, W, 3) -> (B, 3, H, W)
                batch_tensor = torch.tensor(batch / 255.0, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
                # Extract features
                features = self.encoder(batch_tensor)  # Shape: (B, feature_dim)
                all_features.append(features.cpu())
        
        # Concatenate all features
        self.feature_buffer = torch.cat(all_features, dim=0).numpy()
        print(f"[predictor] Feature buffer shape: {self.feature_buffer.shape}")
        print(f"[predictor] Feature buffer construction complete!")
        print(f"============= Feature Buffer Ready =============")

    def load_checkpoint(self):
        """Load Mobipi 3DGS scene model and setup camera configuration."""
        
        scene_model_root_dir = self.hydra_cfg.predictor.scene_model_root_dir
        task = self.hydra_cfg.env.name
        layout = self.unwrapped_env.layout_id
        style = self.unwrapped_env.style_id
        seed = self.unwrapped_env.seed
        
        print(f"\n============= Mobipi Predictor: Loading 3DGS Scene Model =============")
        
        if not os.path.exists(f"{self.hydra_cfg.paths.root}/scene_data"):
            os.system(f"ln -s {scene_model_root_dir} {self.hydra_cfg.paths.root}/scene_data")


        # Find scene model checkpoint path
        scene_model_regex = os.path.join(
            "scene_data",
            camel_to_snake_case(task),
            f"layout{layout}_style{style}_seed{seed}",
            "model/splatfacto/*/nerfstudio_models/",
            "step-*.ckpt",
        )
        print(f"[predictor] Searching for scene model: {scene_model_regex}")
        
        matching_paths = sorted(glob(scene_model_regex))
        if not matching_paths:
            raise FileNotFoundError(f"No scene model found matching: {scene_model_regex}")
        
        scene_model_path = matching_paths[-1]  # Use latest checkpoint
        # scene_model_dir is 2 levels up from the ckpt file
        scene_model_dir = os.path.dirname(os.path.dirname(scene_model_path))
        print(f"[predictor] Found scene model checkpoint: {scene_model_path}")
        print(f"[predictor] Scene model directory: {scene_model_dir}")
        
        # Determine camera names based on task
        # Following eval_mobipi.py Line 266-269
        if task == "CloseDrawer":
            self.camera_names = ["robot0_agentview_right", "robot0_agentview_left"]
        else:
            self.camera_names = ["robot0_agentview_right"]
        
        print(f"[predictor] Using cameras: {self.camera_names}")
        
        # Get camera intrinsics from environment
        camera_intrinsics_list = []
        for camera_name in self.camera_names:
            intrinsics = get_camera_intrinsic_matrix(
                self.unwrapped_env.sim,
                camera_name,
                camera_height=self.image_size,
                camera_width=self.image_size,
            )
            camera_intrinsics_list.append(intrinsics)
            # Convert to dict format required by BatchSceneModel
        self.camera_intrinsics_dict = [
            {
                "w": self.image_size,
                "h": self.image_size,
                "fl_x": float(intrinsics[0, 0]),
                "fl_y": float(intrinsics[1, 1]),
                "cx": float(intrinsics[0, -1]),
                "cy": float(intrinsics[1, -1]),
            }
            for intrinsics in camera_intrinsics_list
        ]
        
        # Load BatchSceneModel
        self.batch_scene_model = BatchSceneModel(scene_model_dir, self.camera_intrinsics_dict)
        print(f"[predictor] BatchSceneModel loaded successfully!")
        
        # Get relative camera poses (camera position/orientation relative to robot base)
        print(f"[predictor] Computing relative camera poses...")
        self.rel_cam_positions = []
        self.rel_cam_mats = []
        for camera_name in self.camera_names:
            rel_pos, rel_mat = compute_relative_cam_pose(
                self.unwrapped_env, camera_name=camera_name
            )
            self.rel_cam_positions.append(torch.tensor(rel_pos, dtype=torch.float32, device=self.device))
            self.rel_cam_mats.append(torch.tensor(rel_mat, dtype=torch.float32, device=self.device))
        print(f"============= 3DGS Scene Model Loading Complete =============")
    
    def _get_robot_images_and_masks(self):
        """Get robot images and masks from current robot position.
        
        Following eval_mobipi.py Line 372-395.
        """
        print(f"\n============= Getting Robot Images and Masks =============")
        
        robot_imgs = []
        robot_masks = []
        
        for camera_name in self.camera_names:
            # Render image with robot at current position
            rendered_image = self.unwrapped_env.sim.render(
                camera_name=camera_name,
                width=self.image_size,
                height=self.image_size,
            )
            
            # Render segmentation mask
            rendered_seg = self.unwrapped_env.sim.render(
                camera_name=camera_name,
                width=self.image_size,
                height=self.image_size,
                depth=False,
                segmentation=True,
            )
            
            # Extract robot geom IDs
            robot_geoms = [
                (geom_id, geom_name)
                for geom_id, geom_name in enumerate(self.unwrapped_env.sim.model.geom_names)
                if "robot0" in geom_name
            ]
            robot_geom_ids = [geom_id for geom_id, geom_name in robot_geoms]
            
            # Create robot mask
            robot_mask = rendered_seg[..., 1].copy()
            robot_mask[~np.isin(rendered_seg[..., 1], robot_geom_ids)] = 0  # Non-robot pixels
            robot_mask[np.isin(rendered_seg[..., 1], robot_geom_ids)] = 1   # Robot pixels
            
            # Flip images (robosuite renders upside down)
            rendered_image = rendered_image[::-1]
            robot_mask = robot_mask[::-1]
            
            robot_imgs.append(rendered_image)
            robot_masks.append(robot_mask)
            
            print(f"[predictor] Camera {camera_name}: robot mask covers {robot_mask.sum()} pixels")
        
        # Convert to torch tensors
        self.robot_imgs_torch = torch.tensor(
            np.array(robot_imgs) / 255.0,
            dtype=torch.float32,
            device=self.device,
        )  # Shape: (num_cameras, H, W, 3)
        
        self.robot_masks_torch = torch.tensor(
            np.array(robot_masks)[..., None],
            dtype=torch.float32,
            device=self.device,
        )  # Shape: (num_cameras, H, W, 1)
        
        print(f"[predictor] Robot images shape: {self.robot_imgs_torch.shape}")
        print(f"[predictor] Robot masks shape: {self.robot_masks_torch.shape}")
        print(f"============= Robot Images and Masks Ready =============")
    
    @property
    def name(self) -> str:
        """Return predictor name."""
        return "mobipi"
