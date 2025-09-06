def visualize_attention(pts, center, attn_weights):
    """
    Visualizes attention weights on the point cloud.
    
    Args:
        pts (Tensor): Original input point cloud (B, N_pts, 3).
        center (Tensor): Center points of the point groups (B, N_groups, 3).
        attn_weights (Tensor): Attention matrix from the final block (B, num_heads, N, N).
    """
    # Assuming batch size is 1 for simplicity
    attn = attn_weights[0]  # Shape: (num_heads, N, N)
    
    # Average attention across all heads for simplicity, or pick a specific head
    attn_avg = attn.mean(dim=0)  # Shape: (N, N)

    # Get the attention scores of the class token (index 0) to all other tokens
    # Note: The first token is the cls token, subsequent tokens correspond to point groups
    cls_attn = attn_avg[0, 1:]  # Shape: (N_groups,)
    
    # Normalize scores for better visualization (e.g., between 0 and 1)
    cls_attn_normalized = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
    
    # You can now use a library like Matplotlib, Mayavi, or Open3D to visualize
    # the points. For each point in `center`, use `cls_attn_normalized` to
    # determine its color (e.g., a colormap from blue to red) or size.
    
    # Here's a conceptual example using a library for visualization
    # Let's say we have a function `plot_points_with_colors`
    # You would need to implement this part based on your chosen library.
    
    # For visualization, you'd plot `center` points. The `cls_attn_normalized`
    # provides the scalar value for the heatmap.
    
    # Example for coloring points:
    # colors = plt.cm.jet(cls_attn_normalized.numpy())[:, :3] # Get RGB from colormap
    # plot_points_with_colors(center.numpy()[0], colors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Assuming you have loaded your model and data
    model = PointTransformer(config)
    model.load_checkpoint(bert_ckpt_path)

    # Prepare a single ego-centric point cloud for visualization
    # This should be a point cloud from your dataset.
    input_pts = ... # Your point cloud data, shape (1, N_pts, 3)

    # Run the forward pass to get the output and attention weights
    output_features, attn_weights = model(input_pts)

    # Get the center points from the group divider (you might need to run it again)
    _, center = model.group_divider(input_pts)

    # Call the visualization function
    visualize_attention(input_pts, center, attn_weights)