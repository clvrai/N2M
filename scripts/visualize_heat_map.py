import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import argparse
import json
import torch
import time

def se2_heatmap_with_arrows_gmm(pcl_numpy, means, covs, weights, grid_size=0.02, arrow_stride=2, extent=0.5):
    # 1. Grid setup
    start_time = time.time()
    x_range = np.arange(means[0][0] - extent, means[0][0] + extent, grid_size)
    y_range = np.arange(means[0][1] - extent, means[0][1] + extent, grid_size)
    theta_range = np.linspace(-np.pi, np.pi, 100)

    xx, yy = np.meshgrid(x_range, y_range)
    heat_vals = np.zeros_like(xx)
    arrow_dirs = np.zeros_like(xx)
    print(f"Grid setup time: {time.time() - start_time} seconds")

    # 2. GMM distribution over [x, y, θ]
    start_time = time.time()
    # Create meshgrid of positions
    pos_x = xx.reshape(-1, 1)
    pos_y = yy.reshape(-1, 1)
    pos_theta = theta_range.reshape(1, -1)
    
    # Create array of all positions (N_points, N_theta, 3)
    positions = np.zeros((len(pos_x), len(theta_range), 3))
    positions[:, :, 0] = pos_x
    positions[:, :, 1] = pos_y 
    positions[:, :, 2] = pos_theta

    # Calculate densities for all positions and components in parallel
    densities = np.zeros((len(pos_x), len(theta_range)))
    for mean, cov, weight in zip(means, covs, weights):
        densities += weight * multivariate_normal.pdf(positions, mean=mean, cov=cov)

    # Sum over theta dimension and reshape back to grid
    heat_vals = np.sum(densities, axis=1).reshape(xx.shape)
    
    # Get theta value with max density for each point
    arrow_dirs = theta_range[np.argmax(densities, axis=1)].reshape(xx.shape)

    # Normalize heatmap for colormap
    heat_vals_norm = (heat_vals - np.min(heat_vals)) / (np.max(heat_vals) - np.min(heat_vals) + 1e-9)
    max_density = np.max(heat_vals)
    density_threshold = 0.1 * max_density
    print(f"GMM distribution time: {time.time() - start_time} seconds")

    # 3. Visualize with Open3D
    geometries = []
    start_time = time.time()
    for i in range(0, xx.shape[0], arrow_stride):
        for j in range(0, xx.shape[1], arrow_stride):
            if heat_vals[i, j] < density_threshold:
                continue  # Skip low-density points
            x, y = xx[i, j], yy[i, j]
            z = 1.0
            theta = arrow_dirs[i, j]

            # Arrow direction
            dx = 0.05 * np.cos(theta)
            dy = 0.05 * np.sin(theta)

            # Arrow base and tip
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.001,
                cone_radius=0.003,
                cylinder_height=0.02,
                cone_height=0.01
            )
            arrow.paint_uniform_color([0.0, 0.0, 0.0])
            # arrow.paint_uniform_color(plt.cm.jet(heat_vals_norm[i, j])[:3])
            
            # First rotate 90 degrees around y axis to point in +x direction
            R_init = arrow.get_rotation_matrix_from_axis_angle([0, np.pi/2, 0])
            arrow.rotate(R_init)
            
            arrow.translate((x, y, z))

            # Align arrow to direction (dx, dy)
            rot_angle = np.arctan2(dy, dx)
            R = arrow.get_rotation_matrix_from_axis_angle([0, 0, rot_angle])
            arrow.rotate(R, center=(x, y, z))

            geometries.append(arrow)

    # Add a colored grid plane (squares) with opacity
    plane_meshes = []
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            if heat_vals[i, j] < density_threshold:
                continue  # Skip low-density points
            x, y = xx[i, j], yy[i, j]
            z = 1.0 - 0.001  # Slightly below the arrows
            color = plt.cm.jet(heat_vals_norm[i, j])[:3]
            square = o3d.geometry.TriangleMesh.create_box(width=grid_size, height=grid_size, depth=0.001)
            square.translate((x - grid_size/2, y - grid_size/2, z))
            square.paint_uniform_color(color)
            plane_meshes.append(square)
    geometries.extend(plane_meshes)
    
    # Convert arrow geometries to point clouds
    point_clouds = []
    for geometry in geometries:
        if isinstance(geometry, o3d.geometry.TriangleMesh):
            # Sample points from the mesh surface
            pcd = geometry.sample_points_uniformly(number_of_points=100)
            point_clouds.append(pcd)
        else:
            point_clouds.append(geometry)
    
    # Replace geometries list with point clouds
    geometries = point_clouds
    
    # Convert point cloud to small spheres
    # spheres = []
    # for point in pcl_numpy[:, :3]:
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    #     sphere.translate(point)
    #     if pcl_numpy.shape[1] > 3:  # If color information exists
    #         sphere.paint_uniform_color(pcl_numpy[0, 3:6])  # Use first point's color
    #     else:
    #         sphere.paint_uniform_color([1, 0, 0])  # Red for visibility
    #     spheres.append(sphere)
    # geometries.extend(spheres)
        
    # Create point cloud geometry
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl_numpy[:, :3])
    if pcl_numpy.shape[1] > 3:  # If color information exists
        pcd.colors = o3d.utility.Vector3dVector(pcl_numpy[:, 3:6])
    geometries.append(pcd)

    print(f"Visualization time: {time.time() - start_time} seconds")
    
    # Visualize all geometries
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sir_config", type=str, required=True)
    parser.add_argument("--pcl_path", type=str, required=True)
    args = parser.parse_args()

    from nav2man.model.SIRPredictor import SIRPredictor
    from nav2man.utils.point_cloud import fix_point_cloud_size
    
    with open(args.sir_config, "r") as f:
        SIR_config = json.load(f)
    
    pcd = o3d.io.read_point_cloud(args.pcl_path)
    pcl_numpy = np.asarray(pcd.points)
    if pcd.has_colors():
        pcl_numpy = np.concatenate([pcl_numpy, np.asarray(pcd.colors)], axis=1)
    pcl_input = fix_point_cloud_size(pcl_numpy, SIR_config['dataset']['pointnum'])
    pcl_input = torch.from_numpy(pcl_input)[None, ...].cuda().float()

    model_config = SIR_config["model"]
    SIR_predictor = SIRPredictor(config=model_config)
    SIR_predictor.eval()
    SIR_predictor.cuda()
    with torch.no_grad():
        means, covs, weights = SIR_predictor(pcl_input)
        means = means[0].cpu().numpy()
        covs = covs[0].cpu().numpy()
        weights = weights[0].cpu().numpy()

    # Example usage
    se2_heatmap_with_arrows_gmm(pcl_numpy, means, covs, weights)