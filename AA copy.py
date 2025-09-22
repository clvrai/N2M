import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import transforms

def load_pcd(pcd_path):
    """Load PCD file and return point cloud data"""
    pcd = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(pcd.points)

def create_occupancy_grid(points, resolution=0.1, ground_z=0.5):
    """Create occupancy grid from point cloud, filtering out ground points below ground_z"""
    # Filter out ground points
    non_ground_points = points[points[:, 2] >= ground_z]
    
    # Get min and max coordinates
    min_coords = np.min(non_ground_points, axis=0)
    max_coords = np.max(non_ground_points, axis=0)
    
    # Calculate grid dimensions
    width = int(np.ceil((max_coords[0] - min_coords[0]) / resolution))
    height = int(np.ceil((max_coords[1] - min_coords[1]) / resolution))
    
    # Initialize occupancy grid
    occupancy_grid = np.zeros((height, width))
    
    # Project points to grid
    for point in non_ground_points:
        x_idx = int((point[0] - min_coords[0]) / resolution)
        y_idx = int((point[1] - min_coords[1]) / resolution)
        
        if 0 <= x_idx < width and 0 <= y_idx < height:
            occupancy_grid[y_idx, x_idx] = 1
    
    return occupancy_grid, min_coords, max_coords, resolution

def check_rectangle_occupancy(occupancy_grid, min_coords, resolution, se2_pose, length=0.630, width=0.500):
    """Check if all grids intersected by the rectangle are empty"""
    x, y, theta = se2_pose
    
    # Get rectangle corners in world frame
    corners = np.array([
        [-length*0.83, -width/2],          # Bottom left
        [length*0.17, -width/2],           # Bottom right
        [length*0.17, width/2],            # Top right
        [-length*0.83, width/2]            # Top left
    ])
    
    # Rotate corners
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    rotated_corners = (rot_matrix @ corners.T).T
    
    # Translate corners
    world_corners = rotated_corners + np.array([x, y])
    
    # Get min and max coordinates of the rectangle
    rect_min = np.min(world_corners, axis=0)
    rect_max = np.max(world_corners, axis=0)
    
    # Convert to grid indices
    min_x_idx = int((rect_min[0] - min_coords[0]) / resolution)
    min_y_idx = int((rect_min[1] - min_coords[1]) / resolution)
    max_x_idx = int((rect_max[0] - min_coords[0]) / resolution) + 1
    max_y_idx = int((rect_max[1] - min_coords[1]) / resolution) + 1
    
    # Ensure indices are within grid bounds
    min_x_idx = max(0, min_x_idx)
    min_y_idx = max(0, min_y_idx)
    max_x_idx = min(occupancy_grid.shape[1], max_x_idx)
    max_y_idx = min(occupancy_grid.shape[0], max_y_idx)
    
    # Check each grid cell in the bounding box
    for y_idx in range(min_y_idx, max_y_idx):
        for x_idx in range(min_x_idx, max_x_idx):
            # Get grid cell corners in world coordinates
            grid_min_x = min_coords[0] + x_idx * resolution
            grid_min_y = min_coords[1] + y_idx * resolution
            grid_max_x = grid_min_x + resolution
            grid_max_y = grid_min_y + resolution
            
            # Check if grid cell intersects with rectangle
            # Convert grid corners to rectangle's local frame
            grid_corners = np.array([
                [grid_min_x, grid_min_y],
                [grid_max_x, grid_min_y],
                [grid_max_x, grid_max_y],
                [grid_min_x, grid_max_y]
            ])
            
            # Translate and rotate grid corners to rectangle's local frame
            translated_corners = grid_corners - np.array([x, y])
            local_corners = (rot_matrix.T @ translated_corners.T).T
            
            # Check if any grid corner is inside rectangle
            # Rectangle bounds in local frame
            rect_min_x = -length*0.83
            rect_max_x = length*0.17
            rect_min_y = -width/2
            rect_max_y = width/2
            
            # Check if any grid corner is inside rectangle
            for corner in local_corners:
                if (rect_min_x <= corner[0] <= rect_max_x and 
                    rect_min_y <= corner[1] <= rect_max_y):
                    if occupancy_grid[y_idx, x_idx] == 1:
                        return False
                    break
            
            # Also check if rectangle corners are inside grid cell
            for corner in world_corners:
                if (grid_min_x <= corner[0] <= grid_max_x and 
                    grid_min_y <= corner[1] <= grid_max_y):
                    if occupancy_grid[y_idx, x_idx] == 1:
                        return False
                    break
    
    return True

def draw_rectangle(ax, se2_pose, length=0.630, width=0.500):
    """Draw a rectangle with given SE2 pose, origin at center of width edge"""
    x, y, theta = se2_pose
    
    # Create rectangle with origin at center of width edge
    # The rectangle extends from (0, -width/2) to (length, width/2)
    rect = patches.Rectangle((-length*0.83, -width/2), length, width, 
                            linewidth=2, edgecolor='r', facecolor='none')
    
    # Create transformation
    t = transforms.Affine2D().rotate(theta).translate(x, y)
    rect.set_transform(t + ax.transData)
    
    # Add rectangle to plot
    ax.add_patch(rect)
    
    # Draw coordinate frame
    arrow_length = 0.2
    # X-axis (red)
    ax.arrow(x, y, 
             arrow_length * np.cos(theta), 
             arrow_length * np.sin(theta), 
             head_width=0.05, head_length=0.05, fc='r', ec='r')
    # Y-axis (green)
    ax.arrow(x, y, 
             arrow_length * np.cos(theta + np.pi/2), 
             arrow_length * np.sin(theta + np.pi/2), 
             head_width=0.05, head_length=0.05, fc='g', ec='g')

def visualize_occupancy_grid(occupancy_grid, min_coords, max_coords, resolution, se2_pose):
    """Visualize occupancy grid using matplotlib"""
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # Plot occupancy grid
    plt.imshow(occupancy_grid, cmap='binary', origin='lower',
              extent=[min_coords[0], max_coords[0], min_coords[1], max_coords[1]])
    
    # Draw rectangle with SE2 pose
    draw_rectangle(ax, se2_pose, length=0.630, width=0.500)
    
    # Check if rectangle area is free
    is_free = check_rectangle_occupancy(occupancy_grid, min_coords, resolution, se2_pose)
    print(f"Rectangle area is {'free' if is_free else 'occupied'}")
    
    # Add axis labels
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add title
    plt.title('Occupancy Grid Map with Rectangle')
    
    # Show plot
    plt.show()

def main():
    # Load PCD file
    # pcd_path = "pcd/0.pcd"
    pcd_path = "pcd/43.pcd"
    points = load_pcd(pcd_path)
    
    # Create occupancy grid (resolution in meters)
    resolution = 0.02  # You can adjust this value
    ground_z = 0.05    # Ground height threshold in meters
    import time
    start_time = time.time()
    occupancy_grid, min_coords, max_coords, resolution = create_occupancy_grid(points, resolution, ground_z)
    end_time = time.time()
    print(f"Time taken to create occupancy grid: {(end_time - start_time)*1000} ms")
    
    # Visualize occupancy grid
    se2_pose = [2.4, -0.9, 1.9]  # [x, y, theta] in meters and radians
    # se2_pose = [2., -1.0, 1.57]  # [x, y, theta] in meters and radians
    visualize_occupancy_grid(occupancy_grid, min_coords, max_coords, resolution, se2_pose)

if __name__ == "__main__":
    main()