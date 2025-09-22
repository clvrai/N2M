import argparse
import open3d as o3d
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os

from n2m.model.N2Mnet import N2Mnet
from n2m.utils.point_cloud import fix_point_cloud_size

def visualize_feature_similarity(original_pts, center, feature_vectors, similarity_metric='cosine'):
    """
    Visualizes feature similarity on the point cloud by mapping group-level feature similarity 
    back to individual points and coloring them based on similarity scores with the CLS token.
    
    Args:
        original_pts (np.array): Original full-resolution point cloud (N_pts, 6) - [x, y, z, r, g, b].
        center (Tensor): Center points of the point groups (B, N_groups, 3).
        feature_vectors (Tensor): Feature vectors from the encoder (B, N_tokens, feature_dim).
        similarity_metric (str): Similarity metric to use ('cosine', 'euclidean', 'dot_product').
    """
    # Convert tensors to numpy for processing
    center_np = center[0].cpu().numpy()  # Shape: (N_groups, 3)
    features_np = feature_vectors[0].cpu().numpy()  # Shape: (N_tokens, feature_dim)
    
    # Debug prints
    print(f"Debug - Feature vectors shape: {features_np.shape}")
    print(f"Debug - Center shape: {center_np.shape}")
    print(f"Debug - Original points shape: {original_pts.shape}")
    
    # Extract coordinates and colors from original point cloud
    points_xyz = original_pts[:, :3]  # (N_pts, 3)
    original_colors = original_pts[:, 3:6]  # (N_pts, 3) - RGB values
    
    # Check if we have the expected structure (CLS token + group tokens)
    expected_tokens = center_np.shape[0] + 1  # N_groups + 1 CLS token
    if features_np.shape[0] != expected_tokens:
        raise ValueError(f"Expected {expected_tokens} tokens (1 CLS + {center_np.shape[0]} groups), but got {features_np.shape[0]} tokens")
    
    # Extract CLS token features and group token features
    # Token 0 is CLS, tokens 1 to N_groups are the group tokens
    cls_features = features_np[0]  # Shape: (feature_dim,)
    group_features = features_np[1:]  # Shape: (N_groups, feature_dim)
    
    print(f"Debug - CLS features shape: {cls_features.shape}")
    print(f"Debug - Group features shape: {group_features.shape}")
    print(f"Debug - Expected {center_np.shape[0]} group features, got {group_features.shape[0]}")
    
    if group_features.shape[0] == 0:
        raise ValueError("No group features found. The feature vector seems to only contain the CLS token.")
    
    # Calculate similarity between CLS token and each group token
    if similarity_metric == 'cosine':
        # Cosine similarity
        cls_norm = np.linalg.norm(cls_features)
        group_norms = np.linalg.norm(group_features, axis=1)
        dot_products = np.dot(group_features, cls_features)
        similarities = dot_products / (cls_norm * group_norms + 1e-8)  # Add small epsilon to avoid division by zero
    elif similarity_metric == 'euclidean':
        # Negative Euclidean distance (higher values = more similar)
        distances = np.linalg.norm(group_features - cls_features[None, :], axis=1)
        similarities = -distances
    elif similarity_metric == 'dot_product':
        # Dot product similarity
        similarities = np.dot(group_features, cls_features)
    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")
    
    # Normalize similarity scores between 0 and 1
    if similarities.max() > similarities.min():
        similarities_normalized = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    else:
        similarities_normalized = np.ones_like(similarities)
    
    # Map group-level similarity back to individual points
    # Use nearest neighbor search to find which group each point belongs to
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(center_np)
    distances, indices = nbrs.kneighbors(points_xyz)
    
    # Assign similarity scores to each point based on its nearest group center
    point_similarity_scores = similarities_normalized[indices.flatten()]
    
    # Create a colormap for visualization (blue = low similarity, red = high similarity)
    cmap = plt.cm.jet
    colors_similarity = cmap(point_similarity_scores)[:, :3]  # Get RGB values
    
    # Create Open3D point cloud for visualization
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points_xyz)
    pcd_original.colors = o3d.utility.Vector3dVector(original_colors / 255.0 if original_colors.max() > 1 else original_colors)
    
    pcd_similarity = o3d.geometry.PointCloud()
    pcd_similarity.points = o3d.utility.Vector3dVector(points_xyz)
    pcd_similarity.colors = o3d.utility.Vector3dVector(colors_similarity)
    
    # Create group centers visualization
    pcd_centers = o3d.geometry.PointCloud()
    pcd_centers.points = o3d.utility.Vector3dVector(center_np)
    center_colors = cmap(similarities_normalized)[:, :3]
    pcd_centers.colors = o3d.utility.Vector3dVector(center_colors)
    
    # Make centers larger by creating spheres
    spheres = []
    for i, center_point in enumerate(center_np):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(center_point)
        sphere.paint_uniform_color(center_colors[i])
        spheres.append(sphere)
    
    print(f"Feature similarity ({similarity_metric}) - Min: {similarities.min():.4f}, Max: {similarities.max():.4f}, Mean: {similarities.mean():.4f}")
    print(f"Normalized similarity - Min: {similarities_normalized.min():.4f}, Max: {similarities_normalized.max():.4f}, Mean: {similarities_normalized.mean():.4f}")
    print(f"Number of points: {len(points_xyz)}")
    print(f"Number of groups: {len(center_np)}")
    print(f"Feature dimension: {cls_features.shape[0]}")
    
    # Save point clouds instead of visualizing
    print("Saving point clouds as PLY files...")
    os.makedirs("vis", exist_ok=True)
    o3d.io.write_point_cloud("vis/original_rgb.ply", pcd_original)
    o3d.io.write_point_cloud("vis/similarity_colored.ply", pcd_similarity)
    # Save group centers as a point cloud (not spheres)
    o3d.io.write_point_cloud("vis/group_centers.ply", pcd_centers)

    pcd_similarity_translated = pcd_similarity.translate((0, -3.0, 0))  # Translate for better visualization
    combined = pcd_original + pcd_similarity_translated
    o3d.io.write_point_cloud("vis/combined.ply", combined)
    print("Saved: original_rgb.ply, similarity_colored.ply, group_centers.ply, combined.ply")
    return point_similarity_scores, similarities_normalized

def load_pcd_file(pcd_path, pointnum=8192):
    """
    Load a PCD file and return the downsampled point cloud for model inference.
    
    Args:
        pcd_path (str): Path to the PCD file.
        pointnum (int): Number of points to downsample to.
        
    Returns:
        np.array: Point cloud data with shape (pointnum, 6) - [x, y, z, r, g, b]
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)  # (N, 3)
    colors = np.asarray(pcd.colors)  # (N, 3)
    
    # Combine points and colors
    if colors.size > 0:
        # If colors are in [0, 1] range, convert to [0, 255]
        if colors.max() <= 1.0:
            colors = colors * 255.0
        point_cloud = np.concatenate([points, colors], axis=1)  # (N, 6)
    else:
        # If no colors, use white as default
        colors = np.ones((points.shape[0], 3)) * 255.0
        point_cloud = np.concatenate([points, colors], axis=1)  # (N, 6)

    # downsample to specified number of points
    point_cloud = fix_point_cloud_size(point_cloud, pointnum)
    
    return point_cloud

def load_original_pcd_file(pcd_path):
    """
    Load the original PCD file without downsampling for visualization.
    
    Args:
        pcd_path (str): Path to the PCD file.
        
    Returns:
        np.array: Original point cloud data with shape (N, 6) - [x, y, z, r, g, b]
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)  # (N, 3)
    colors = np.asarray(pcd.colors)  # (N, 3)
    
    # Combine points and colors
    if colors.size > 0:
        # If colors are in [0, 1] range, convert to [0, 255]
        if colors.max() <= 1.0:
            colors = colors * 255.0
        point_cloud = np.concatenate([points, colors], axis=1)  # (N, 6)
    else:
        # If no colors, use white as default
        colors = np.ones((points.shape[0], 3)) * 255.0
        point_cloud = np.concatenate([points, colors], axis=1)  # (N, 6)
    
    return point_cloud

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize feature similarity on RGB point clouds")
    parser.add_argument("--config", type=str, required=True, help="Path to model config JSON file")
    parser.add_argument("--pcd", type=str, required=True, help="Path to PCD file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to run the model on")
    parser.add_argument("--similarity", type=str, default="cosine", choices=["cosine", "euclidean", "dot_product"],
                       help="Similarity metric to use for comparing features")
    parser.add_argument("--model_points", type=int, default=8192,
                       help="Number of points to use for model inference (default: 8192)")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)
    
    print(f"Loading model with config: {args.config}")
    print(f"Using device: {args.device}")
    print(f"Using similarity metric: {args.similarity}")
    
    # Load and initialize the model
    model = N2Mnet(config['model'])
    model = model.to(args.device)
    model.eval()
    encoder = model.encoder
    
    # Load point cloud data
    print(f"Loading point cloud: {args.pcd}")
    
    # Load original full-resolution point cloud for visualization
    original_point_cloud_np = load_original_pcd_file(args.pcd)
    print(f"Loaded original point cloud with {original_point_cloud_np.shape[0]} points")
    
    # Load downsampled point cloud for model inference
    point_cloud_np = load_pcd_file(args.pcd, pointnum=args.model_points)
    print(f"Loaded downsampled point cloud with {point_cloud_np.shape[0]} points for model inference")
    
    # Convert downsampled points to tensor and add batch dimension for model
    input_pts = torch.from_numpy(point_cloud_np).float().unsqueeze(0).to(args.device)  # (1, N, 6)
    
    print("Running forward pass...")
    with torch.no_grad():
        # Check if the encoder uses max pooling
        original_use_max_pool = encoder.use_max_pool
        
        # Temporarily disable max pooling to get individual token features
        encoder.use_max_pool = False
        feature_vectors, neighborhood, center = encoder.forward_neighbohood_center(input_pts)
        
        # Restore original setting
        encoder.use_max_pool = original_use_max_pool
        
        print(f"Feature vectors shape: {feature_vectors.shape}")
        print(f"Neighborhood shape: {neighborhood.shape}")
        print(f"Center shape: {center.shape}")
    
    print("Visualizing feature similarity...")
    # Call the visualization function with original point cloud
    point_scores, group_scores = visualize_feature_similarity(
        original_point_cloud_np, center, feature_vectors, 
        similarity_metric=args.similarity
    )
    
    print("\nVisualization complete!")
    print(f"Point-level similarity scores range: [{point_scores.min():.4f}, {point_scores.max():.4f}]")
    print(f"Group-level similarity scores range: [{group_scores.min():.4f}, {group_scores.max():.4f}]")