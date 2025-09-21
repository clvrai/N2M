import open3d as o3d
from open3d.visualization import draw_geometries
import numpy as np
import sys
import os
import argparse
import glob

def visualize_pcd(pcd_path):
    """
    Visualize a PCD file using Open3D
    
    Args:
        pcd_path (str): Path to the PCD file
    """
    # Check if file exists
    if not os.path.exists(pcd_path):
        print(f"Error: File {pcd_path} does not exist.")
        return False
    
    # Load the point cloud
    print(f"Loading point cloud from {pcd_path}...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # Print basic information about the point cloud
    print(f"Point cloud loaded with {len(np.asarray(pcd.points))} points.")
    print(f"Dimension: {np.asarray(pcd.points).shape}")
    
    # Add colors if not present
    if not pcd.has_colors():
        print("Point cloud has no colors, adding colors based on coordinates...")
        points = np.asarray(pcd.points)
        colors = np.zeros((len(points), 3))
        # Normalize coordinates to [0,1] range for coloring
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        range_vals = max_vals - min_vals
        
        # Avoid division by zero
        range_vals[range_vals == 0] = 1
        
        # Use XYZ for RGB coloring
        norm_pts = (points - min_vals) / range_vals
        colors[:, 0] = norm_pts[:, 0]  # R from X
        colors[:, 1] = norm_pts[:, 1]  # G from Y
        colors[:, 2] = norm_pts[:, 2]  # B from Z
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    # Visualize the point cloud with custom options
    print("Visualizing point cloud. Press 'Q' to exit.")
    o3d.visualization.draw_geometries([pcd, coordinate_frame])  # type: ignore
    
    return True

def get_latest_pcd_id(directory):
    """
    Get the file with the highest ID in the directory
    
    Args:
        directory (str): Directory to search for PCD files
    
    Returns:
        int: The highest ID found
    """
    pcd_files = glob.glob(os.path.join(directory, "*.pcd"))
    if not pcd_files:
        print(f"No PCD files found in {directory}")
        return None
    
    ids = []
    for file_path in pcd_files:
        filename = os.path.basename(file_path)
        try:
            # Extract ID from filename (assuming format like "23.pcd")
            file_id = int(os.path.splitext(filename)[0])
            ids.append(file_id)
        except ValueError:
            # Skip files that don't have numeric IDs
            continue
    
    if not ids:
        print(f"No files with numeric IDs found in {directory}")
        return None
    
    return max(ids)

if __name__ == "__main__":
    # Default directory path
    
    # default_directory = "n2m_inference/chair/prediction"
    # default_directory = "dataset_rollout/handover_15/pcl"
    default_directory = "dataset_rollout/lamp_3/pcl"
    # default_directory = "n2m_inference/microwave/prediction"
    # default_directory = "dataset_rollout/handover_6/pcl"
    # default_directory = "dataset_rollout/microwave/prediction"
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Visualize PCD files")
    parser.add_argument("directory", nargs="?", default=default_directory,
                        help=f"Directory containing PCD files (default: {default_directory})")
    parser.add_argument("--id", type=int, 
                        help="ID of the PCD file to visualize (default: latest ID). Use -1 to stitch all PCD files.")
    
    args = parser.parse_args()
    
    # Make sure directory exists
    if not os.path.isdir(args.directory):
        print(f"Error: Directory {args.directory} does not exist or is not a directory")
        sys.exit(1)
    
    if args.id == -1:
        # Stitch all PCD files in the directory
        pcd_files = sorted(glob.glob(os.path.join(args.directory, "*.pcd")))
        if not pcd_files:
            print(f"No PCD files found in {args.directory}")
            sys.exit(1)
        print(f"Stitching {len(pcd_files)} PCD files...")
        all_points = []
        all_colors = []
        for pcd_file in pcd_files:
            pcd = o3d.io.read_point_cloud(pcd_file)
            all_points.append(np.asarray(pcd.points))
            if pcd.has_colors():
                all_colors.append(np.asarray(pcd.colors))
            else:
                all_colors.append(None)
        # Concatenate all points
        points = np.vstack(all_points)
        # Handle colors: if any file has no color, fallback to None for all
        if all(c is not None for c in all_colors):
            colors = np.vstack(all_colors)
        else:
            colors = None
        # Create stitched point cloud
        stitched_pcd = o3d.geometry.PointCloud()
        stitched_pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            stitched_pcd.colors = o3d.utility.Vector3dVector(colors)
        # Save to a temp file or visualize directly
        print("Visualizing stitched point cloud. Press 'Q' to exit.")
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        o3d.visualization.draw_geometries([stitched_pcd, coordinate_frame])  # type: ignore
        sys.exit(0)
    
    # If ID is specified, use it; otherwise find the latest ID
    if args.id is not None:
        pcd_id = args.id
    else:
        pcd_id = get_latest_pcd_id(args.directory)
        if pcd_id is None:
            print(f"No valid PCD files found in {args.directory}")
            sys.exit(1)
        print(f"Using latest PCD file with ID: {pcd_id}")
    
    # Construct the full path to the PCD file
    pcd_path = os.path.join(args.directory, f"{pcd_id}.pcd")
    
    # Visualize the PCD file
    visualize_pcd(pcd_path)
