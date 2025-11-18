"""Collision detection utilities.

Migrated from A_ref/N2M-sim/robomimic/robomimic/utils/sample_utils.py
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Optional


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
        ground_z: float = 0.05
    ):
        """Initialize collision checker.
        
        Args:
            point_cloud: Point cloud of the scene
            resolution: Grid resolution in meters
            robot_width: Robot width in meters
            robot_length: Robot length in meters  
            ground_z: Ground plane z height
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
