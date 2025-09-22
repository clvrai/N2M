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
    
    T_base_cams = [
        [
            [-4.79426771e-01, -9.03563376e-05,  8.77581897e-01,  8.12595641e-02],
            [-4.33193158e-05,  1.00000000e+00,  7.92950479e-05,  3.00092773e-02],
            [-8.77581893e-01, -7.90657298e-11, -4.79426763e-01,  1.41216779e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ], # top
        [
            [-4.79512557e-01, -9.40567361e-05,  8.77535013e-01,  6.47147592e-02],
            [-4.51053399e-05,  9.99999995e-01,  8.25359144e-05,  3.00090098e-02],
            [-8.77535017e-01, -4.48849964e-09, -4.79512559e-01,  1.26911545e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ], # middle
        [
            [-4.79427762e-01,  1.42015666e-04,  8.77581336e-01,  5.36461060e-02],
            [ 6.80864395e-05,  9.99999990e-01, -1.24630215e-04,  2.99852649e-02],
            [-8.77581344e-01,  2.11597055e-10, -4.79427767e-01,  1.05299580e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ] # bottom
    ]
    camera_intrinsic = [500, 500, 320, 260, 640, 520]

    print("loading meta...")
    meta_path = os.path.join(args.dataset_path, "rollout", "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    episodes = meta['episodes']
    print("meta loaded")

    camera_pose_save_dir = os.path.join(args.dataset_path, "rollout", "camera_poses")
    os.makedirs(camera_pose_save_dir, exist_ok=True)

    if args.vis:
        camera_pose_vis_dir = os.path.join(args.dataset_path, "rollout", "camera_poses_vis")
        os.makedirs(camera_pose_vis_dir, exist_ok=True)

    camera_poses = []
    for episode in tqdm(episodes):
        print(f"processing {episode['id']}")
        pcl_path = os.path.join(args.dataset_path, 'rollout', episode['file_path'])
        pcl = o3d.io.read_point_cloud(pcl_path)
        se2_origin = episode['meta_info']['se2_origin']
        furniture_pos = episode['meta_info']['furniture_pos']
        furniture_pos = np.array(furniture_pos)

        print(f"loading target helper")
        target_helper = TargetHelper(pcl, se2_origin, x_half_range=1, y_half_range=1.5, theta_half_range_deg=90, vis=False, camera_intrinsic=camera_intrinsic)
        poses = []
        for _ in range(args.num_poses):
            print(f"sampling {_} / {args.num_poses}")
            height_index = np.random.randint(0, 3)
            target_helper.T_base_cam = T_base_cams[height_index]
            print(f"height index: {height_index}")
            base_se2, camera_extrinsic = target_helper.get_random_target_se2_with_boundary_check(furniture_pos[:2])
            camera_extrinsic[0, 3] += 0.5
            poses.append(camera_extrinsic.tolist())

        if args.vis:
            save_pose_visualization(pcl, poses, furniture_pos, os.path.join(camera_pose_vis_dir, f"{episode['id']}.pcd"))

        camera_poses.append(poses)
        
    with open(os.path.join(camera_pose_save_dir, "camera_poses.json"), "w") as f:
        json.dump(camera_poses, f)
