import numpy as np
import open3d as o3d
import torch
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
import yaml


def _load_point_cloud(file_path, use_color=True):
    pcd = o3d.io.read_point_cloud(file_path)
    point_cloud = np.asarray(pcd.points)

    if use_color:
        colors = np.asarray(pcd.colors)
        point_cloud = np.concatenate([point_cloud, colors], axis=1)
    return point_cloud

def gmm_pdf(points, means, covs, weights):
    pdf_vals = np.zeros(len(points))
    for k in range(len(weights)):
        mvn = multivariate_normal(mean=means[k], cov=covs[k])
        pdf_vals += weights[k] * mvn.pdf(points)
    return pdf_vals

def create_ellipsoid(mean, cov, color, scale=1.0, num_points=100000):
    """
    Create an ellipsoid from a mean and covariance matrix
    
    Args:
        mean: Mean vector (3,)
        cov: Covariance matrix (3, 3)
        num_points: Number of points to generate for the ellipsoid
    """
    # Eigen-decomposition of covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Create a sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere.compute_vertex_normals()
    
    # Scale sphere to form ellipsoid
    scales = scale * np.sqrt(eigvals)
    sphere_vertices = np.asarray(sphere.vertices)
    sphere_vertices = sphere_vertices @ np.diag(scales)
    
    # Rotate according to eigenvectors
    sphere_vertices = sphere_vertices @ eigvecs.T
    sphere.vertices = o3d.utility.Vector3dVector(sphere_vertices)
    
    # Translate to mean
    sphere.translate(mean)
    
    # Color the ellipsoid
    sphere.paint_uniform_color(color)
    
    return sphere

def visualize_gmm_distribution(point_cloud, target_point, label, means, covs, weights):
    """
    Visualize GMM distribution as a 3D heatmap overlaid on point cloud
    
    Args:
        point_cloud: Input point cloud (N, 3)
        means: Mean vectors for each Gaussian component (K, 3)
        covs: Covariance matrices for each Gaussian component (K, 3, 3)
        weights: Mixing coefficients for each Gaussian component (K)
        num_points: Number of points to generate for the heatmap
        true_points: List of true points to visualize as spheres (optional)
    """
    
    # Create point cloud object for input points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
    
    # Create ellipsoids for each Gaussian component
    ellipsoids = []
    for i in range(len(weights)):
        ellipsoid = create_ellipsoid(means[i], covs[i], color=[0, 1, 0], scale=2.0)
        ellipsoids.append(ellipsoid)
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0])
    
    # Combine point clouds
    geometries = [pcd, coord_frame] + ellipsoids
    
    # Add spheres for true points if provided
    if target_point is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(target_point)

        if label == 1:
            sphere.paint_uniform_color([0, 0, 1])  # Green color for true points
        else:
            sphere.paint_uniform_color([1, 0, 0])  # Red color for false points
        geometries.append(sphere)
    
    # Return all geometries
    return geometries

def save_gmm_visualization(point_cloud, target_point, label, means, covs, weights, output_path):
    """
    Save GMM visualization to a file
    
    Args:
        point_cloud: Input point cloud (N, 3)
        means: Mean vectors for each Gaussian component (K, 3)
        covs: Covariance matrices for each Gaussian component (K, 3, 3)
        weights: Mixing coefficients for each Gaussian component (K)
        output_path: Path to save the visualization
        num_points: Number of points to generate for the heatmap
        true_points: List of true points to visualize as spheres (optional)
    """
    geometries = visualize_gmm_distribution(
        point_cloud, target_point, label, means, covs, weights
    )
    
    # Extract combined point cloud and convert other geometries to points
    combined_points = []
    combined_colors = []
    
    # Add the main point cloud points and colors
    combined_points.append(np.asarray(geometries[0].points))
    combined_colors.append(np.asarray(geometries[0].colors))
    
    # Convert coordinate frame to points
    coord_mesh = geometries[1]
    coord_pcd = coord_mesh.sample_points_uniformly(number_of_points=1000)
    combined_points.append(np.asarray(coord_pcd.points))
    combined_colors.append(np.asarray(coord_pcd.colors))
    
    # Convert ellipsoids to points if they exist
    for ellipsoid in geometries[2:]:
        ellipsoid_pcd = ellipsoid.sample_points_uniformly(number_of_points=10000)
        combined_points.append(np.asarray(ellipsoid_pcd.points))
        combined_colors.append(np.asarray(ellipsoid_pcd.colors))
    
    # Combine all points and colors
    all_points = np.vstack(combined_points)
    all_colors = np.vstack(combined_colors)

    all_colors = np.clip(all_colors, 0, 1)
    
    # Create final point cloud
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(all_points)
    final_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # Save to file
    o3d.io.write_point_cloud(output_path, final_pcd)


def save_gmm_visualization_se2(point_cloud, target_se2, label, means, covs, weights, output_path, interval=0.1, area=[-10, 10, -10, 10], threshold=0.5, z_value=0.8):
    """
    Given a point cloud and mean and covariance of SE(2) points, visualize the point cloud and the SE(2) points.
    For each grid points in the xy area with interval, calculate the most probable theta
    And then with SE(2) point (x, y, theta), calculate the probalility of that SE(2) point
    Based on the probability, color the grid and visualize the arrow of theta.
    Repeat this process for all grid points and save this with the point cloud.

    Args:
        point_cloud: Input point cloud (N, 3)
        target_se2: Target SE(2) point to visualize
        label: Label of the target point (0 or 1)
        means: Mean vectors for each Gaussian component (K, 3)
        covs: Covariance matrices for each Gaussian component (K, 3, 3)
        weights: Mixing coefficients for each Gaussian component (K)
        interval: Interval of the grid to visualize the SE(2) points
        area: Area to visualize the SE(2) points
        threshold: Threshold of the probability to visualize the SE(2) points

    Returns:
        no returns, save the visualization to output_path. 
    """
    # Create a grid of points in the xy area
    x = np.arange(area[0], area[1] + interval, interval)
    y = np.arange(area[2], area[3] + interval, interval)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    
    # Initialize arrays to store results
    n_grid_points = len(grid_points)
    theta_samples = np.linspace(-np.pi, np.pi, 36)  # Sample 36 different angles
    
    # Create all possible combinations of grid points and thetas
    # Shape: (n_grid_points * n_thetas, 3)
    grid_expanded = np.repeat(grid_points, len(theta_samples), axis=0)
    theta_expanded = np.tile(theta_samples, n_grid_points)
    all_se2_points = np.column_stack((grid_expanded, theta_expanded))
    
    # Calculate probabilities for all points at once
    all_probs = gmm_pdf(all_se2_points, means, covs, weights)
    
    # Reshape probabilities to (n_grid_points, n_thetas)
    probs_matrix = all_probs.reshape(n_grid_points, len(theta_samples))
    
    # Find the best theta and probability for each grid point
    best_probs = np.max(probs_matrix, axis=1)
    best_theta_indices = np.argmax(probs_matrix, axis=1)
    best_thetas = theta_samples[best_theta_indices]
    
    # Filter out probabilities below threshold and normalize probabilities to [0,1]
    prob_min, prob_max = best_probs.min(), best_probs.max()
    best_probs = (best_probs - prob_min) / (prob_max - prob_min)
    
    # Create visualization geometries
    geometries = []
    
    # Add original point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
    geometries.append(pcd)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    geometries.append(coord_frame)
    
    # Add target SE(2) point if provided
    if target_se2 is not None:
        # Create sphere for position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=interval/4)
        sphere.translate([target_se2[0], target_se2[1], z_value])  # Use x,y coordinates
        if label == 1:
            sphere.paint_uniform_color([0, 0, 1])  # Blue for positive label
        else:
            sphere.paint_uniform_color([1, 0, 0])  # Red for negative label
        geometries.append(sphere)

        # Create arrow for theta
        arrow_length = interval * 1
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=interval/16,
            cone_radius=interval/12,
            cylinder_height=arrow_length*0.8,
            cone_height=arrow_length*0.2
        )
        
        # Rotate and translate arrow to position
        R_x = np.array([
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]
            ])
        R = np.array([
            [np.cos(target_se2[2]), -np.sin(target_se2[2]), 0],
            [np.sin(target_se2[2]), np.cos(target_se2[2]), 0],
            [0, 0, 1]
        ])
        arrow.rotate(R_x, center=[0, 0, 0]) # rotate to x-axis
        arrow.rotate(R, center=[0, 0, 0]) # rotate to theta
        arrow.translate([target_se2[0], target_se2[1], z_value])
        arrow.paint_uniform_color([0, 0, 1] if label == 1 else [1, 0, 0])
        geometries.append(arrow)
    
    # Add grid points and arrows for points above threshold
    arrow_length = interval * 1
    for i in range(len(best_probs)):
        if best_probs[i] > threshold:
            # Create sphere for grid point
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=interval/8)
            sphere.translate([grid_points[i,0], grid_points[i,1], z_value])
            
            # Color based on probability (blue->red gradient)
            red_value = (best_probs[i] - threshold) / (1 - threshold) * 0.99
            color = [1-red_value, red_value, 0]
            sphere.paint_uniform_color(color)
            geometries.append(sphere)
            
            # Create arrow for theta
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=interval/16,
                cone_radius=interval/12,
                cylinder_height=arrow_length*0.8,
                cone_height=arrow_length*0.2
            )
            
            # Rotate and translate arrow to position
            R_x = np.array([
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]
            ])
            R = np.array([
                [np.cos(best_thetas[i]), -np.sin(best_thetas[i]), 0],
                [np.sin(best_thetas[i]), np.cos(best_thetas[i]), 0],
                [0, 0, 1]
            ])
            arrow.rotate(R_x, center=[0, 0, 0]) # rotate to x-axis
            arrow.rotate(R, center=[0, 0, 0]) # rotate to theta
            arrow.translate([grid_points[i,0], grid_points[i,1], z_value])
            arrow.paint_uniform_color(color)
            geometries.append(arrow)
    
    # Convert all geometries to point clouds and combine
    combined_points = []
    combined_colors = []
    
    for geom in geometries:
        if isinstance(geom, o3d.geometry.PointCloud):
            combined_points.append(np.asarray(geom.points))
            combined_colors.append(np.asarray(geom.colors))
        else:
            # Sample points from mesh
            pcd = geom.sample_points_uniformly(number_of_points=500)
            combined_points.append(np.asarray(pcd.points))
            combined_colors.append(np.asarray(pcd.colors))
    
    # Combine all points and colors
    all_points = np.vstack(combined_points)
    all_colors = np.vstack(combined_colors)
    all_colors = np.clip(all_colors, 0, 1)
    
    # Create final point cloud
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(all_points)
    final_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # Save to file
    o3d.io.write_point_cloud(output_path, final_pcd)

def save_gmm_visualization_xythetaz(point_cloud, target_xythetaz, label, means, covs, weights, output_path, interval=0.1, area=[-10, 10, -10, 10, -10, 10], threshold=0.5):
    """
    Given a point cloud and mean and covariance of SE(2)+Z points, visualize the point cloud and the SE(2)+Z points.
    For each grid points in the xyz area with interval, calculate the most probable theta
    And then with SE(2)+Z point (x, y, theta, z), calculate the probability of that point
    Based on the probability, color the grid and visualize the arrow of theta.
    Repeat this process for all grid points and save this with the point cloud.

    Args:
        point_cloud: Input point cloud (N, 3)
        target_xythetaz: Target SE(2)+Z point to visualize (x,y,theta,z)
        label: Label of the target point (0 or 1)
        means: Mean vectors for each Gaussian component (K, 4)
        covs: Covariance matrices for each Gaussian component (K, 4, 4)
        weights: Mixing coefficients for each Gaussian component (K)
        interval: Interval of the grid to visualize the points
        area: Area to visualize the points [xmin, xmax, ymin, ymax, zmin, zmax]
        threshold: Threshold of the probability to visualize the points
    """
    # Create a grid of points in the xyz area
    x = np.arange(area[0], area[1] + interval, interval)
    y = np.arange(area[2], area[3] + interval, interval)
    z = np.arange(area[4], area[5] + interval, interval)
    X, Y, Z = np.meshgrid(x, y, z)
    grid_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Initialize arrays to store results
    n_grid_points = len(grid_points)
    theta_samples = np.linspace(-np.pi, np.pi, 36)  # Sample 36 different angles
    
    # Create all possible combinations of grid points and thetas
    # Shape: (n_grid_points * n_thetas, 4)
    grid_expanded = np.repeat(grid_points, len(theta_samples), axis=0)
    theta_expanded = np.tile(theta_samples, n_grid_points)
    # Rearrange to x,y,theta,z format
    all_xythetaz_points = np.column_stack((
        grid_expanded[:, 0],  # x
        grid_expanded[:, 1],  # y
        theta_expanded,       # theta
        grid_expanded[:, 2]   # z
    ))
    
    # Calculate probabilities for all points at once
    all_probs = gmm_pdf(all_xythetaz_points, means, covs, weights)
    
    # Reshape probabilities to (n_grid_points, n_thetas)
    probs_matrix = all_probs.reshape(n_grid_points, len(theta_samples))
    
    # Find the best theta and probability for each grid point
    best_probs = np.max(probs_matrix, axis=1)
    best_theta_indices = np.argmax(probs_matrix, axis=1)
    best_thetas = theta_samples[best_theta_indices]
    
    # Filter out probabilities below threshold and normalize probabilities to [0,1]
    prob_min, prob_max = best_probs.min(), best_probs.max()
    best_probs = (best_probs - prob_min) / (prob_max - prob_min)
    
    # Create visualization geometries
    geometries = []
    
    # Add original point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
    geometries.append(pcd)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    geometries.append(coord_frame)
    
    # Add target xythetaz point if provided
    if target_xythetaz is not None:
        # Create sphere for position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=interval/4)
        sphere.translate([target_xythetaz[0], target_xythetaz[1], target_xythetaz[3]])  # Use x,y,z coordinates
        if label == 1:
            sphere.paint_uniform_color([0, 0, 1])  # Blue for positive label
        else:
            sphere.paint_uniform_color([1, 0, 0])  # Red for negative label
        geometries.append(sphere)

        # Create arrow for theta
        arrow_length = interval * 1
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=interval/16,
            cone_radius=interval/12,
            cylinder_height=arrow_length*0.8,
            cone_height=arrow_length*0.2
        )
        
        # Rotate and translate arrow to position
        R_x = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        R = np.array([
            [np.cos(target_xythetaz[2]), -np.sin(target_xythetaz[2]), 0],
            [np.sin(target_xythetaz[2]), np.cos(target_xythetaz[2]), 0],
            [0, 0, 1]
        ])
        arrow.rotate(R_x, center=[0, 0, 0]) # rotate to x-axis
        arrow.rotate(R, center=[0, 0, 0]) # rotate to theta
        arrow.translate([target_xythetaz[0], target_xythetaz[1], target_xythetaz[3]])
        arrow.paint_uniform_color([0, 0, 1] if label == 1 else [1, 0, 0])
        geometries.append(arrow)
    
    # Add grid points and arrows for points above threshold
    arrow_length = interval * 1
    for i in range(len(best_probs)):
        if best_probs[i] > threshold:
            # Create sphere for grid point
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=interval/8)
            sphere.translate([grid_points[i,0], grid_points[i,1], grid_points[i,2]])
            
            # Color based on probability (blue->red gradient)
            red_value = (best_probs[i] - threshold) / (1 - threshold) * 0.99
            color = [1-red_value, red_value, 0]
            sphere.paint_uniform_color(color)
            geometries.append(sphere)
            
            # Create arrow for theta
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=interval/16,
                cone_radius=interval/12,
                cylinder_height=arrow_length*0.8,
                cone_height=arrow_length*0.2
            )
            
            # Rotate and translate arrow to position
            R_x = np.array([
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]
            ])
            R = np.array([
                [np.cos(best_thetas[i]), -np.sin(best_thetas[i]), 0],
                [np.sin(best_thetas[i]), np.cos(best_thetas[i]), 0],
                [0, 0, 1]
            ])
            arrow.rotate(R_x, center=[0, 0, 0]) # rotate to x-axis
            arrow.rotate(R, center=[0, 0, 0]) # rotate to theta
            arrow.translate([grid_points[i,0], grid_points[i,1], grid_points[i,2]])
            arrow.paint_uniform_color(color)
            geometries.append(arrow)
    
    # Convert all geometries to point clouds and combine
    combined_points = []
    combined_colors = []
    
    for geom in geometries:
        if isinstance(geom, o3d.geometry.PointCloud):
            combined_points.append(np.asarray(geom.points))
            combined_colors.append(np.asarray(geom.colors))
        else:
            # Sample points from mesh
            pcd = geom.sample_points_uniformly(number_of_points=500)
            combined_points.append(np.asarray(pcd.points))
            combined_colors.append(np.asarray(pcd.colors))
    
    # Combine all points and colors
    all_points = np.vstack(combined_points)
    all_colors = np.vstack(combined_colors)
    all_colors = np.clip(all_colors, 0, 1)
    
    # Create final point cloud
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(all_points)
    final_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # Save to file
    o3d.io.write_point_cloud(output_path, final_pcd)

if __name__ == "__main__":
    # Load data
    point_cloud = _load_point_cloud("/home/hyunjun/projects/CoRL2025/nav2man/exp_pcl_train/data/point_clouds/0001.ply")
    means = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    covs = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
    weights = [0.1, 0.4, 0.5]
    save_gmm_visualization(point_cloud, means, covs, weights, "data/gmm_visualization")