import argparse
import json
import time
from collections import OrderedDict
import os
import numpy as np
import robosuite
import open3d as o3d
from scipy.spatial.transform import Rotation

from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from termcolor import colored
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map, transform_from_pixels_to_world

from robocasa.models.scenes.scene_registry import LayoutType, StyleType
from robocasa.scripts.collect_demos import collect_human_trajectory

from termcolor import colored
import cv2

def choose_option(
    options, option_name, show_keys=False, default=None, default_message=None
):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all tasks

    if default is None:
        default = options[0]

    if default_message is None:
        default_message = default

    # Select environment to run
    print("{}s:".format(option_name.capitalize()))

    for i, (k, v) in enumerate(options.items()):
        if show_keys:
            print("[{}] {}: {}".format(i, k, v))
        else:
            print("[{}] {}".format(i, v))
    print()
    try:
        s = input(
            "Choose an option 0 to {}, or any other key for default ({}): ".format(
                len(options) - 1,
                default_message,
            )
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except:
        if default is None:
            choice = options[0]
        else:
            choice = default
        print("Use {} by default.\n".format(choice))

    # Return the chosen environment name
    return choice


def create_point_cloud_from_rgbd(color_img, depth_img, intrinsic, extrinsic=np.eye(4)):
    """
    Convert RGB-D images to colored point cloud using direct back-projection
    
    Args:
        color_img (ndarray): RGB image of shape (H, W, 3)
        depth_img (ndarray): Depth image of shape (H, W, 1) or (H, W) with real distances
        intrinsic (ndarray): Camera intrinsic matrix (3x3)
        extrinsic (ndarray): Camera extrinsic matrix (4x4), defaults to identity
        
    Returns:
        o3d.geometry.PointCloud: Colored point cloud
    """
    
    # Ensure depth is in the right shape (H, W, 1)
    if len(depth_img.shape) == 2:
        depth_img = depth_img[..., None]
    elif len(depth_img.shape) == 3 and depth_img.shape[2] == 1:
        pass
    else:
        print(f"Warning: Unexpected depth image shape: {depth_img.shape}")
        if len(depth_img.shape) > 2:
            depth_img = depth_img[:, :, 0:1]  # Take first channel
    
    # Ensure RGB is 3-channel (H, W, 3)
    if len(color_img.shape) != 3 or color_img.shape[2] != 3:
        print(f"Warning: Unexpected RGB image shape: {color_img.shape}")
        if len(color_img.shape) == 2:  # If grayscale, convert to RGB
            color_img = np.stack([color_img] * 3, axis=-1)
    
    try:
        # Get image dimensions
        rows, cols = depth_img.shape[:2]
        
        # Create pixel coordinates grid
        y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        pixels = np.stack([x, y], axis=-1)  # shape (H, W, 2)
        
        # Filter invalid depth values
        # VITAL: maximum depth is 20.0
        valid = np.isfinite(depth_img) & (depth_img > 0.0) & (depth_img < 20.0)
        valid = valid[..., 0]  # Remove last dimension
        
        # print(f"Number of valid points: {np.sum(valid)}")
        
        if np.sum(valid) == 0:
            print("Warning: No valid depth points found")
            return o3d.geometry.PointCloud()
        
        # Get valid pixels and corresponding colors
        valid_pixels = pixels[valid]
        valid_colors = color_img[valid] / 255.0  # Normalize to 0-1 range
        valid_depths = depth_img[valid]  # Get only valid depth values
        
        # Convert pixel coordinates to camera coordinates
        # First, subtract principal point and divide by focal length
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        # Convert to camera coordinates
        x_cam = (valid_pixels[:, 0] - cx) * valid_depths[:, 0] / fx
        y_cam = (valid_pixels[:, 1] - cy) * valid_depths[:, 0] / fy
        z_cam = valid_depths[:, 0]
        
        # Stack camera coordinates
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        # Transform from camera to world coordinates
        # First convert to homogeneous coordinates
        points_cam_hom = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
        
        # Apply extrinsic transformation
        points_world_hom = np.dot(extrinsic, points_cam_hom.T).T
        points_world = points_world_hom[:, :3]
        
        # print(f"Transformed points shape: {points_world.shape}")
        # print(f"Transformed points range: min={np.min(points_world, axis=0)}, max={np.max(points_world, axis=0)}")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(valid_colors)
        
        # Filter out outliers if enough points
        if len(points_world) > 100:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
        return pcd
    
    except Exception as e:
        print(f"Error creating point cloud: {e}")
        import traceback
        traceback.print_exc()
        # Return an empty point cloud
        return o3d.geometry.PointCloud()


def capture_depth_camera_data(env, camera_name='None', save_dir=None, id=None):
    """
    Capture RGB, depth images and pointcloud from depth_camera using robosuite camera utilities
    
    Args:
        env: Environment instance
        save_dir (str): Directory to save captures
    """
    sim = env.sim
    available_cameras = [sim.model.camera_id2name(i) for i in range(sim.model.ncam)]
    
    if camera_name not in available_cameras:
        print("Error: depth_camera not in simulator")
        print(colored(f"Available cameras: {available_cameras}", "red"))
        return
    
    # Get current timestamp for unique filenames
    timestamp = int(time.time())
    
    # Get the observations (this forces the renderer to update with latest scene)
    obs = env._get_observations(force_update=True)
    
    # Get camera parameters using robosuite utilities
    camera_height = env.camera_heights[env.camera_names.index(camera_name)]
    camera_width = env.camera_widths[env.camera_names.index(camera_name)]
    
    # Get camera matrices using robosuite utilities
    intrinsic_matrix = get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width)
    extrinsic_matrix = get_camera_extrinsic_matrix(sim, camera_name)

    
    # Get RGB and depth images
    rgb_img = obs.get(f"{camera_name}_image")
    depth_img = obs.get(f"{camera_name}_depth")
    
    # Flip images upside down
    rgb_img = np.flipud(rgb_img)
    depth_img = np.flipud(depth_img)
    
    if rgb_img is None or depth_img is None:
        print("Error: Could not get RGB or depth images from depth_camera")
        return
    
    # Convert depth to real distances using robosuite utility
    depth_img = get_real_depth_map(sim, depth_img)
    
    # Create and save point cloud using the camera matrices
    pcd = create_point_cloud_from_rgbd(rgb_img, depth_img, intrinsic_matrix, extrinsic_matrix)

    if save_dir is not None and id is not None:
        # Save RGB/depth/intrinsic_matrix image
        cv2.imwrite(os.path.join(save_dir, "img", f"{id}_{camera_name}.png"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        
        # use np.save to save depth_img as numpy array, ensure no precision loss
        os.makedirs(os.path.join(save_dir, "depth"), exist_ok=True)
        np.save(os.path.join(save_dir, "depth", f"{id}_{camera_name}.npy"), depth_img)
        
        # intrinsic_matrix to [fx, fy, cx, cy, width, height]
        intrinsic_list = [intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2], camera_width, camera_height]
        intrinsic_extrinsic_matrix_dict = {
            "intrinsic_matrix": intrinsic_list,
            "extrinsic_matrix": extrinsic_matrix.tolist(),
        }
        with open(os.path.join(save_dir, "info", f"{id}_{camera_name}.json"), "w") as f:
            json.dump(intrinsic_extrinsic_matrix_dict, f,  indent=4)
        
        # (normalized for visualization)
        # depth_min = np.min(depth_img)
        # depth_max = np.max(depth_img)
        # depth_viz = (depth_img - depth_min) / (depth_max - depth_min) if depth_max > depth_min else np.zeros_like(depth_img)
        # depth_viz = (depth_viz * 255).astype(np.uint8)
        
        # # Save raw depth values
        # depth_raw_filename = os.path.join(save_dir, f"depth_raw_{timestamp}.npy")
        # np.save(depth_raw_filename, depth_img)
        
        # pcd_filename = os.path.join(save_dir, f"pointcloud.ply")
        # o3d.io.write_point_cloud(pcd_filename, pcd)
    return pcd


def debug_env_cameras(env):
    """
    Debug function to print all available cameras and observation keys
    
    Args:
        env: Environment instance
    """
    print("\n--- DEBUG INFORMATION ---")
    print("Environment camera names:")
    print(env.camera_names)
    
    # Get all camera names from the simulator
    sim = env.sim
    available_cameras = [sim.model.camera_id2name(i) for i in range(sim.model.ncam)]
    print("\nAll cameras in simulator:")
    print(available_cameras)
    
    # Print camera parameters for each camera if available
    print("\nCamera parameters:")
    for i in range(sim.model.ncam):
        cam_name = sim.model.camera_id2name(i)
        print(f"Camera: {cam_name}")
        print(f"  Position: {sim.model.cam_pos[i]}")
        print(f"  Quaternion: {sim.model.cam_quat[i]}")
        print(f"  FOV: {sim.model.cam_fovy[i]}")
    
    # Get observations and print keys
    obs = env._get_observations(force_update=True)
    print("\nObservation keys:")
    print(list(obs.keys()))
    
    # Print camera-related observation keys
    camera_obs = [key for key in obs.keys() if "camera" in key or "image" in key or "depth" in key]
    print("\nCamera-related observation keys:")
    print(camera_obs)
    
    print("\nModalities:")
    print(env.observation_modalities)
    
    print("\nCamera depth settings:")
    print(f"Camera depths: {env.camera_depths}")
    print("--- END DEBUG INFORMATION ---\n")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PnPCounterToCab", help="task")
    parser.add_argument("--layout", type=int, help="kitchen layout (choose number 0-9)")
    parser.add_argument("--style", type=int, help="kitchen style (choose number 0-11)")
    parser.add_argument("--robot", type=str, help="robot", default="PandaOmron")
    parser.add_argument("--capture", action="store_true", help="capture data from depth camera")
    parser.add_argument("--savedir", type=str, default="depth_camera_captures", help="directory to save captures")
    parser.add_argument("--debug", action="store_true", help="print debug information about cameras and observations")
    args = parser.parse_args()
    args.capture = True
    raw_layouts = dict(
        map(lambda item: (item.value, item.name.lower().capitalize()), LayoutType)
    )
    layouts = OrderedDict()
    for k in sorted(raw_layouts.keys()):
        if k < -0:
            continue
        layouts[k] = raw_layouts[k]

    raw_styles = dict(
        map(lambda item: (item.value, item.name.lower().capitalize()), StyleType)
    )
    styles = OrderedDict()
    for k in sorted(raw_styles.keys()):
        if k < 0:
            continue
        styles[k] = raw_styles[k]

    # Create argument configuration
    config = {
        "env_name": args.task,
        "robots": args.robot,
        "controller_configs": load_composite_controller_config(robot=args.robot),
        "translucent_robot": False,
    }

    args.renderer = "mjviewer"

    print(colored("Initializing environment...", "yellow"))

    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,  # Need this for camera captures
        render_camera=None,
        ignore_done=True,
        use_camera_obs=True,  # Need this for camera captures
        control_freq=20,
        renderer=args.renderer,
        camera_names=["depth_camera1", "depth_camera2", "depth_camera3", "depth_camera4", "depth_camera5", "robot0_front_depth"],  # Fixed camera name
        camera_heights=[240, 240, 240, 240, 240, 240],
        camera_widths=[320, 320, 320, 320, 320, 320],
        camera_depths=[True, True, True, True, True, True],  # depth_camera provides depth
        layout_and_style_ids=[[args.layout, args.style]],
    )

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # initialize device
    from robosuite.devices import Keyboard

    device = Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)

    # collect demonstrations
    while True:
        if args.layout is None:
            layout = choose_option(
                layouts, "kitchen layout", default=-1, default_message="random layouts"
            )
        else:
            layout = args.layout

        if args.style is None:
            style = choose_option(
                styles, "kitchen style", default=-1, default_message="random styles"
            )
        else:
            style = args.style

        if layout == -1:
            layout = np.random.choice(range(10))
        if style == -1:
            style = np.random.choice(range(11))

        env.layout_and_style_ids = [[layout, style]]
        print(
            colored(
                f"Showing configuration:\n    Layout: {layouts[layout]}\n    Style: {styles[style]}",
                "green",
            )
        )
        print()
        print(
            colored(
                "Spawning environment...\n(Press Q any time to view new configuration)",
                "yellow",
            )
        )
        
        # Reset the environment to make sure camera is initialized
        env.reset()

        # Debug information if requested
        if args.debug:
            debug_env_cameras(env)

        # Capture data from depth camera if requested
        if args.capture:
            print(colored("Capturing data from depth camera...", "cyan"))
            pcd1 = capture_depth_camera_data(env, camera_name='depth_camera1', save_dir=args.savedir)
            pcd2 = capture_depth_camera_data(env, camera_name='depth_camera2', save_dir=args.savedir)
            
            o3d.io.write_point_cloud(os.path.join(args.savedir, "pointcloud.ply"), pcd1+pcd2)
        
        ep_directory, discard_traj = collect_human_trajectory(
            env,
            device,
            "right",
            "single-arm-opposed",
            mirror_actions=True,
            render=(args.renderer != "mjviewer"),
            max_fr=30,
            print_info=True,
        )

        print()
