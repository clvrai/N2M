import open3d as o3d
import numpy as np
import sys
import os
from pathlib import Path
import argparse

def visualize_point_cloud_with_origin(pcd_path):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        print(f"Failed to load point cloud file: {pcd_path}")
        return

    # Create coordinate frame at origin
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )
    coordinate_frame_points = coordinate_frame.sample_points_uniformly(number_of_points=1000)

    # Convert coordinate frame to point cloud
    coord_points = np.asarray(coordinate_frame_points.points)
    coord_colors = np.asarray(coordinate_frame_points.colors)

    # Create point cloud from coordinate frame
    coord_pcd = o3d.geometry.PointCloud()
    coord_pcd.points = o3d.utility.Vector3dVector(coord_points)
    coord_pcd.colors = o3d.utility.Vector3dVector(coord_colors)

    # Combine original point cloud with coordinate frame
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(np.vstack((
        np.asarray(pcd.points),
        np.asarray(coord_pcd.points)
    )))
    combined_pcd.colors = o3d.utility.Vector3dVector(np.vstack((
        np.asarray(pcd.colors),
        np.asarray(coord_pcd.colors)
    )))

    # Save combined point cloud
    output_pcd_path = Path(pcd_path).with_suffix('').with_name(f"{Path(pcd_path).stem}_with_origin.pcd")
    o3d.io.write_point_cloud(str(output_pcd_path), combined_pcd)
    print(f"Saved combined point cloud to: {output_pcd_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd_path", type=str, required=True)
    args = parser.parse_args()
    
    visualize_point_cloud_with_origin(args.pcd_path) 