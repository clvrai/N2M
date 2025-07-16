import argparse
import os
import json
import open3d as o3d
import numpy as np
from tqdm import tqdm

from robomimic.utils.sample_utils import TargetHelper

def save_pose_visualization(pcl, poses, furniture_pos, save_path):
    # Create point cloud for the scene
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl.points)
    pcd.colors = o3d.utility.Vector3dVector(pcl.colors)
    
    # Add furniture position as red points
    furniture_pcd = o3d.geometry.PointCloud()
    furniture_pcd.points = o3d.utility.Vector3dVector([furniture_pos[:3]])
    furniture_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red
    pcd = pcd + furniture_pcd
    
    for pose in poses:
        # Get axis endpoints in world coordinates
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        camera_frame.transform(pose)
        camera_frame_points = camera_frame.sample_points_uniformly(number_of_points=300)
        pcd = pcd + camera_frame_points
    
    # Save combined point cloud
    o3d.io.write_point_cloud(save_path, pcd)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/rollouts/PnPCounterToCab_BCtransformer_rollouts/PnPCounterToCab_BCtransformer_rollout_scene1/20250430230003")
    parser.add_argument("--num_poses", type=int, default=10)
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    print("loading meta...")
    meta_path = os.path.join(args.dataset_path, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    episodes = meta['episodes']
    print("meta loaded")

    camera_pose_save_dir = os.path.join(args.dataset_path, "camera_poses")
    os.makedirs(camera_pose_save_dir, exist_ok=True)

    if args.vis:
        camera_pose_vis_dir = os.path.join(args.dataset_path, "camera_poses_vis")
        os.makedirs(camera_pose_vis_dir, exist_ok=True)

    camera_poses = []
    base_poses = []
    for episode in tqdm(episodes):
        pcl_path = os.path.join(args.dataset_path, episode['file_path'])
        pcl = o3d.io.read_point_cloud(pcl_path)
        se2_origin = episode['meta_info']['se2_origin']
        furniture_pos = episode['meta_info']['furniture_pos']
        furniture_pos = np.array(furniture_pos)

        target_helper = TargetHelper(pcl, se2_origin, x_half_range=2, y_half_range=2, theta_half_range_deg=60, vis=False)
        episode_camera_poses = []
        episode_base_poses = []
        for _ in range(args.num_poses):
            rel_base_se2, camera_extrinsic = target_helper.get_random_target_se2_with_visibility_check(furniture_pos[:2])
            abs_base_se2 = [x + y for x, y in zip(rel_base_se2, se2_origin)]
            matrix_base_se3 = target_helper.calculate_target_se3(abs_base_se2)
            episode_base_poses.append(matrix_base_se3.tolist())
            episode_camera_poses.append(camera_extrinsic.tolist())

        if args.vis:
            save_pose_visualization(pcl, episode_camera_poses, furniture_pos, os.path.join(camera_pose_vis_dir, f"{episode['id']}.pcd"))

        camera_poses.append(episode_camera_poses)
        base_poses.append(episode_base_poses)

    with open(os.path.join(camera_pose_save_dir, "camera_poses.json"), "w") as f:
        json.dump(camera_poses, f, indent=4)
    with open(os.path.join(camera_pose_save_dir, "base_poses.json"), "w") as f:
        json.dump(base_poses, f, indent=4)
