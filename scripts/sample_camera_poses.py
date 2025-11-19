import argparse
import os
import json
import open3d as o3d
import numpy as np
from tqdm import tqdm

from n2m.utils.sample_utils import TargetHelper

def save_pose_visualization(pcl, poses, furniture_pos, robot_se2_pose, save_path):
    # Create point cloud for the scene
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl.points)
    pcd.colors = o3d.utility.Vector3dVector(pcl.colors)
    
    # Add furniture position as red points
    furniture_pcd = o3d.geometry.PointCloud()
    furniture_pcd.points = o3d.utility.Vector3dVector([furniture_pos[:3]])
    furniture_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red
    pcd = pcd + furniture_pcd
    
    # Add camera poses as coordinate frames
    for pose in poses:
        # Get axis endpoints in world coordinates
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        camera_frame.transform(pose)
        camera_frame_points = camera_frame.sample_points_uniformly(number_of_points=300)
        pcd = pcd + camera_frame_points
    
    # Add robot base pose as a blue arrow
    if robot_se2_pose is not None:
        z_value = 0.8  # Height to display the arrow
        interval = 0.2  # Scale for arrow size
        
        # Create sphere for robot position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=interval/4)
        sphere.translate([robot_se2_pose[0], robot_se2_pose[1], z_value])
        sphere.paint_uniform_color([0, 0, 1])  # Blue
        sphere_pcd = sphere.sample_points_uniformly(number_of_points=500)
        pcd = pcd + sphere_pcd
        
        # Create arrow for robot orientation
        arrow_length = interval * 2
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=interval/8,
            cone_radius=interval/6,
            cylinder_height=arrow_length*0.7,
            cone_height=arrow_length*0.3
        )
        
        # Rotate arrow to point along x-axis first, then rotate by theta
        R_x = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        R_theta = np.array([
            [np.cos(robot_se2_pose[2]), -np.sin(robot_se2_pose[2]), 0],
            [np.sin(robot_se2_pose[2]), np.cos(robot_se2_pose[2]), 0],
            [0, 0, 1]
        ])
        arrow.rotate(R_x, center=[0, 0, 0])
        arrow.rotate(R_theta, center=[0, 0, 0])
        arrow.translate([robot_se2_pose[0], robot_se2_pose[1], z_value])
        arrow.paint_uniform_color([0, 0, 1])  # Blue
        arrow_pcd = arrow.sample_points_uniformly(number_of_points=1000)
        pcd = pcd + arrow_pcd
    
    # Save combined point cloud
    o3d.io.write_point_cloud(save_path, pcd)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/rollouts/PnPCounterToCab_BCtransformer_rollouts/PnPCounterToCab_BCtransformer_rollout_scene1/20250430230003")
    parser.add_argument("--num_poses", type=int, default=10)
    parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to process. If not specified, process all episodes.")
    parser.add_argument("--x_half_range", type=float, default=2)
    parser.add_argument("--y_half_range", type=float, default=2)
    parser.add_argument("--theta_half_range_deg", type=float, default=60)
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    print("loading meta...")
    meta_path = os.path.join(args.dataset_path, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    episodes = meta['episodes']
    print("meta loaded")
    
    # Validate and filter episodes based on num_episodes parameter
    if args.num_episodes is not None:
        if len(episodes) < args.num_episodes:
            raise ValueError(f"Error: The meta file contains only {len(episodes)} episodes, but {args.num_episodes} episodes were requested.")
        episodes = episodes[:args.num_episodes]
        print(f"Processing {args.num_episodes} episodes out of {len(meta['episodes'])} total episodes.")

    T_base_to_cam = meta["meta"]["T_base_to_cam"]
    camera_intrinsic = meta["meta"]["camera_intrinsic"]

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
        pose = episode['pose']
        se2_origin = pose

        # load object position if exists, otherwise set it in front of the robot
        if "object_position" in episode:
            object_position = episode['object_position']
        else:
            object_position = pose + [0.5 * np.cos(pose[2]), 0.5 * np.sin(pose[2]), 0]
        object_position = np.array(object_position)

        target_helper = TargetHelper(
            pcl, 
            origin_se2=se2_origin, 
            x_half_range=args.x_half_range, 
            y_half_range=args.y_half_range, 
            theta_half_range_deg=args.theta_half_range_deg, 
            vis=False, 
            camera_intrinsic=camera_intrinsic
        )
        target_helper.T_base_cam = np.array(T_base_to_cam)
        episode_camera_poses = []
        episode_base_poses = []
        for _ in range(args.num_poses):
            rel_base_se2, camera_extrinsic = target_helper.get_random_target_se2_with_visibility_check(object_position[:2])
            abs_base_se2 = [x + y for x, y in zip(rel_base_se2, pose)]
            matrix_base_se3 = target_helper.calculate_target_se3(abs_base_se2)
            episode_base_poses.append(matrix_base_se3.tolist())
            episode_camera_poses.append(camera_extrinsic.tolist())

        if args.vis:
            save_pose_visualization(pcl, episode_camera_poses, object_position, pose, os.path.join(camera_pose_vis_dir, f"{episode['id']}.pcd"))

        camera_poses.append(episode_camera_poses)
        base_poses.append(episode_base_poses)

    with open(os.path.join(camera_pose_save_dir, "camera_poses.json"), "w") as f:
        json.dump(camera_poses, f, indent=4)
    with open(os.path.join(camera_pose_save_dir, "base_poses.json"), "w") as f:
        json.dump(base_poses, f, indent=4)
