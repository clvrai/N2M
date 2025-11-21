"""Collision detection utilities.

Migrated from A_ref/N2M-sim/robomimic/robomimic/utils/sample_utils.py
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from typing import Tuple, Optional
from pathlib import Path


class CollisionChecker:
    """Collision checker for robot base poses.
    
    Uses occupancy grid built from point cloud to check collisions.
    Migrated from TargetHelper class in sample_utils.py
    """
    
    def __init__(
        self, 
        point_cloud: o3d.geometry.PointCloud,
        resolution: float = 0.02,
        robot_width: float = 0.5,
        robot_length: float = 0.63,
        ground_z: float = 0.05,
        camera_intrinsic: Optional[np.ndarray] = None
    ):
        """Initialize collision checker.
        
        Args:
            point_cloud: Point cloud of the scene
            resolution: Grid resolution in meters
            robot_width: Robot width in meters
            robot_length: Robot length in meters  
            ground_z: Ground plane z height
            camera_intrinsic: Camera intrinsic parameters [fx, fy, cx, cy, width, height]
        """
        self.resolution = resolution
        self.width = robot_width
        self.length = robot_length
        self.ground_z = ground_z
        
        # Store point cloud
        self.pcd = point_cloud
        self.pcd_np = np.asarray(self.pcd.points)
        
        # Get point cloud bounds
        self.pcd_max_value = np.max(self.pcd_np, axis=0)
        self.pcd_min_value = np.min(self.pcd_np, axis=0)
        
        # Camera intrinsic for visibility check (following reference line 182-185)
        if camera_intrinsic is not None:
            self.cam_intrinsic = np.array(camera_intrinsic)
        else:
            # Default intrinsic from reference
            self.cam_intrinsic = np.array([100.6919557412736, 100.6919557412736, 160.0, 120.0, 320, 240])
        
        # Base-to-camera transformation (following reference line 156-161)
        self.T_base_cam = np.array([
            [-8.25269110e-02, -5.73057816e-01,  8.15348841e-01,  6.05364230e-04],
            [-9.95784041e-01,  1.45464862e-02, -9.05661474e-02, -3.94417736e-02],
            [ 4.00391906e-02, -8.19385767e-01, -5.71842485e-01,  1.64310488e-00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ])
        
        # Build occupancy grid
        self.occupancy_grid = self._build_occupancy_grid()
        
    def _build_occupancy_grid(self) -> np.ndarray:
        """Build 2D occupancy grid from point cloud."""
        # Calculate grid size
        x_size = int((self.pcd_max_value[0] - self.pcd_min_value[0]) / self.resolution) + 1
        y_size = int((self.pcd_max_value[1] - self.pcd_min_value[1]) / self.resolution) + 1
        
        # Initialize grid (0 = free, 1 = occupied)
        grid = np.zeros((x_size, y_size), dtype=np.uint8)
        
        # Fill grid with points above ground
        for point in self.pcd_np:
            if point[2] > self.ground_z:  # Only consider points above ground
                x_idx = int((point[0] - self.pcd_min_value[0]) / self.resolution)
                y_idx = int((point[1] - self.pcd_min_value[1]) / self.resolution)
                
                # Clamp to grid bounds
                x_idx = max(0, min(x_idx, x_size - 1))
                y_idx = max(0, min(y_idx, y_size - 1))
                
                grid[x_idx, y_idx] = 1
        
        return grid
        
    def check_collision(self, pose: np.ndarray) -> bool:
        """Check if robot at given pose is collision-free.
        
        Following reference implementation semantics:
        Returns True if NO collision (pose is valid), False if collision detected.
        
        Args:
            pose: SE2 pose [x, y, theta]
            
        Returns:
            True if collision-free (pose is valid), False if collision detected
        """
        x, y, theta = pose
        
        # Get rectangle corners in robot frame (following reference: sample_utils.py:357-362)
        # Robot origin is not at center - it's offset towards the back
        corners_robot = np.array([
            [-self.length * 0.83, -self.width / 2],  # Bottom left
            [self.length * 0.17, -self.width / 2],   # Bottom right
            [self.length * 0.17, self.width / 2],    # Top right
            [-self.length * 0.83, self.width / 2]    # Top left
        ])
        
        # Rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R = np.array([[cos_theta, -sin_theta],
                      [sin_theta, cos_theta]])
        
        # Transform corners to world frame
        corners_world = corners_robot @ R.T + np.array([x, y])
        
        # Check all grid cells within bounding box
        min_corner = np.min(corners_world, axis=0)
        max_corner = np.max(corners_world, axis=0)
        
        # Convert to grid indices
        min_x_idx = int((min_corner[0] - self.pcd_min_value[0]) / self.resolution)
        max_x_idx = int((max_corner[0] - self.pcd_min_value[0]) / self.resolution)
        min_y_idx = int((min_corner[1] - self.pcd_min_value[1]) / self.resolution)
        max_y_idx = int((max_corner[1] - self.pcd_min_value[1]) / self.resolution)
        
        # Clamp to grid bounds
        min_x_idx = max(0, min_x_idx)
        max_x_idx = min(self.occupancy_grid.shape[0] - 1, max_x_idx)
        min_y_idx = max(0, min_y_idx)
        max_y_idx = min(self.occupancy_grid.shape[1] - 1, max_y_idx)
        
        # Check if any grid cell within bounding box is occupied
        for i in range(min_x_idx, max_x_idx + 1):
            for j in range(min_y_idx, max_y_idx + 1):
                # Convert grid index to world coordinates
                grid_x = self.pcd_min_value[0] + i * self.resolution
                grid_y = self.pcd_min_value[1] + j * self.resolution
                
                # Check if grid point is inside robot rectangle
                if self._point_in_rectangle(grid_x, grid_y, corners_world):
                    if self.occupancy_grid[i, j] == 1:
                        return False  # Collision detected - pose is invalid
        
        return True  # No collision - pose is valid
    
    def _point_in_rectangle(
        self, 
        px: float, 
        py: float, 
        corners: np.ndarray
    ) -> bool:
        """Check if point is inside rectangle defined by corners.
        
        Uses cross product method.
        """
        # Get vectors
        v0 = corners[1] - corners[0]
        v1 = corners[3] - corners[0]
        v2 = np.array([px, py]) - corners[0]
        
        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        
        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u <= 1) and (v <= 1)
    
    def check_boundary(
        self, 
        pose: np.ndarray, 
        x_range: Tuple[float, float],
        y_range: Tuple[float, float]
    ) -> bool:
        """Check if pose is within specified boundaries.
        
        Args:
            pose: SE2 pose [x, y, theta]
            x_range: (x_min, x_max)
            y_range: (y_min, y_max)
            
        Returns:
            True if within bounds, False otherwise
        """
        x, y, _ = pose
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)
    
    def calculate_target_se3(self, se2_pose: np.ndarray) -> np.ndarray:
        """Convert SE2 pose to SE3 transformation matrix.
        
        Following reference: sample_utils.py line 546-555
        
        Args:
            se2_pose: SE2 pose [x, y, theta]
            
        Returns:
            4x4 SE3 transformation matrix
        """
        x, y, theta = se2_pose
        se3_pose = np.array([
            [np.cos(theta), -np.sin(theta), 0, x],
            [np.sin(theta), np.cos(theta), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return se3_pose
    
    def calculate_camera_extrinsic(self, se2_pose: np.ndarray) -> np.ndarray:
        """Calculate camera extrinsic matrix from robot SE2 pose.
        
        Following reference: sample_utils.py line 557-561
        
        Args:
            se2_pose: Robot SE2 pose [x, y, theta]
            
        Returns:
            4x4 camera extrinsic matrix
        """
        se3_pose = self.calculate_target_se3(se2_pose)
        camera_extrinsic = se3_pose @ self.T_base_cam
        return camera_extrinsic
    
    def check_object_visibility(
        self,
        se2_pose: np.ndarray,
        object_pos: np.ndarray
    ) -> bool:
        """Check if object is visible from camera at given robot pose.
        
        Following reference: sample_utils.py line 563-619
        
        Args:
            se2_pose: Robot SE2 pose [x, y, theta]
            object_pos: Object position [x, y] or [x, y, z]
            
        Returns:
            True if object is visible, False otherwise
        """
        # Get camera extrinsic
        camera_extrinsic = self.calculate_camera_extrinsic(se2_pose)
        
        # Unpack camera intrinsic
        fx, fy, cx, cy, w, h = self.cam_intrinsic
        intrinsic_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # Convert object_pos to homogeneous coordinates (following reference line 574-579)
        if object_pos.shape == (3,):
            object_pos_homo = np.concatenate([object_pos, np.array([1.0])])
        elif object_pos.shape == (2,):
            # Use default height 1.0
            object_pos_homo = np.concatenate([object_pos, np.array([1.0, 1.0])])
        else:
            raise ValueError("object_pos must be a 2D or 3D array")
        
        # Transform object to camera coordinates (following reference line 582-585)
        object_cam = np.linalg.inv(camera_extrinsic) @ object_pos_homo
        object_cam = object_cam[:3] / object_cam[3]
        object_pix = intrinsic_matrix @ object_cam
        object_pix = object_pix[:2] / object_pix[2]
        
        # Check if object is in front of camera (following reference line 612-613)
        if object_cam[2] <= 0:
            return False
        
        # Check if object is within image bounds (following reference line 616-617)
        if object_pix[0] < 0 or object_pix[0] > w or object_pix[1] < 0 or object_pix[1] > h:
            return False
        
        return True
    
    def visualize_occupancy_and_rectangle(
        self,
        se2_initial: np.ndarray,
        sampled_pose: np.ndarray,
        pose_range: dict,
        save_path: Optional[str] = None,
        show_plot: bool = False
    ):
        """Visualize occupancy grid with origin, sampled pose, and sampling region.
        
        Following reference implementation: sample_utils.py line 427-500
        
        Args:
            se2_initial: Origin SE2 pose [x, y, theta]
            sampled_pose: Sampled SE2 pose [x, y, theta]
            pose_range: Dictionary with 'x', 'y', 'theta' ranges
            save_path: Optional path to save figure
            show_plot: Whether to show the plot
        """
        try:
            plt.figure(figsize=(10, 10))
            ax = plt.gca()
            
            # Plot occupancy grid
            # Need to transpose and flip to match coordinates
            extent = [self.pcd_min_value[0], self.pcd_max_value[0], 
                     self.pcd_min_value[1], self.pcd_max_value[1]]
            plt.imshow(self.occupancy_grid.T, cmap='binary', origin='lower', extent=extent)
            
            # Extract pose range
            x_range = pose_range['x']
            y_range = pose_range['y']
            
            # Draw sampling region (blue dashed rectangle)
            x_half_range = (x_range[1] - x_range[0]) / 2.0
            y_half_range = (y_range[1] - y_range[0]) / 2.0
            sampling_rect = patches.Rectangle(
                (se2_initial[0] - x_half_range, se2_initial[1] - y_half_range),
                2 * x_half_range, 2 * y_half_range,
                linewidth=1, edgecolor='b', facecolor='none', linestyle='--', label='Sampling Region'
            )
            ax.add_patch(sampling_rect)
            
            # Draw origin point (blue circle)
            ax.plot(se2_initial[0], se2_initial[1], 'bo', markersize=8, label='Origin')
            
            # Draw origin robot rectangle (green)
            self._draw_robot_rectangle(ax, se2_initial, color='g', label='Origin Robot')
            
            # Draw sampled robot rectangle (red)
            self._draw_robot_rectangle(ax, sampled_pose, color='r', label='Sampled Robot')
            
            # Add axis labels
            plt.xlabel('X (meters)')
            plt.ylabel('Y (meters)')
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            # Add title
            plt.title('Occupancy Grid with Robot Poses')
            
            # Add legend
            plt.legend()
            
            # Save if path provided
            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved visualization to: {save_path}")
            
            # Show plot if requested
            if show_plot:
                plt.show()
                plt.pause(0.5)
            
            plt.close()
            
        except Exception as e:
            print(f"Visualization error: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_robot_rectangle(self, ax, pose: np.ndarray, color: str, label: str):
        """Draw robot rectangle at given pose.
        
        Args:
            ax: Matplotlib axis
            pose: SE2 pose [x, y, theta]
            color: Color for the rectangle
            label: Label for legend
        """
        x, y, theta = pose
        
        # Create rectangle with origin at center of width edge
        # Robot dimensions: length (front-back), width (left-right)
        rect = patches.Rectangle(
            (-self.length * 0.83, -self.width / 2), 
            self.length, 
            self.width, 
            linewidth=2, 
            edgecolor=color, 
            facecolor='none',
            label=label
        )
        
        # Create transformation: rotate then translate
        t = transforms.Affine2D().rotate(theta).translate(x, y)
        rect.set_transform(t + ax.transData)
        
        # Add rectangle to plot
        ax.add_patch(rect)
        
        # Draw coordinate frame arrows
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
