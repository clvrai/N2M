import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.transforms as transforms
import open3d as o3d

def get_target_helper_for_rollout_collection(inference_mode=False, all_pcd=None, se2_origin=None, vis=False, camera_intrinsic=None, filter_noise=True):
    if inference_mode:
        x_half_range = 0.5
        y_half_range = 0.5
        theta_half_range_deg = 30
    else:
        x_half_range = 0.2
        y_half_range = 0.2
        theta_half_range_deg = 15
                
    return TargetHelper(
        all_pcd,
        se2_origin,
        x_half_range,
        y_half_range,
        theta_half_range_deg,
        vis=vis,
        camera_intrinsic=camera_intrinsic,
        filter_noise=filter_noise
    )

class TargetHelper:
    def __init__(self, pcd, origin_se2, x_half_range, y_half_range, theta_half_range_deg, vis=False, camera_intrinsic=None, filter_noise=True):
        self.resolution = 0.02
        self.width = 0.5
        self.length = 0.63
        self.ground_z = 0.05
        # self.T_base_cam = np.array([
        #     [ 5.83445639e-03,  5.87238353e-01, -8.09393072e-01,  4.49474752e-02],
        #     [-9.99982552e-01,  2.64855534e-03, -5.28670366e-03, -1.88012307e-02],
        #     [-9.60832741e-04,  8.09409739e-01,  5.87243519e-01,  1.64493829e-00],
        #     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])  # this is just for DETECT mode and 
        # self.T_base_cam = np.array(
        #     [[-8.14932746e-02, -5.73355454e-01,  8.15243714e-01,  9.75590401e-04],
        #     [-9.95079241e-01,  5.53813432e-04, -9.90804744e-02, -2.56451598e-02],
        #     [ 5.63568407e-02, -8.19306535e-01, -5.70579275e-01,  1.64368076e-01],
        #     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
        # )
        self.T_base_cam = np.array([
            [-8.25269110e-02, -5.73057816e-01,  8.15348841e-01,  6.05364230e-04],
            [-9.95784041e-01,  1.45464862e-02, -9.05661474e-02, -3.94417736e-02],
            [ 4.00391906e-02, -8.19385767e-01, -5.71842485e-01,  1.64310488e-00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ])
        self.arm_length = 1
        
        # Filter noise from point cloud if requested
        if filter_noise:
            self.pcd = self.filter_point_cloud(pcd)
        else:
            self.pcd = pcd
            
        self.origin_se2 = origin_se2
        self.x_half_range = x_half_range
        self.y_half_range = y_half_range
        self.theta_half_range_deg = theta_half_range_deg
        self.to_rad = lambda x: x/180*np.pi
        self.pcd_np = np.asarray(self.pcd.points)
        self.pcd_np = np.concatenate([self.pcd_np, np.zeros((self.pcd_np.shape[0], 1))], axis=1)
        self.pcd_color_np = np.asarray(self.pcd.colors)
        self.pcd_max_value = np.max(self.pcd_np[:, :3], axis=0)
        self.pcd_min_value = np.min(self.pcd_np[:, :3], axis=0)
        self.vis = vis
        
        if camera_intrinsic is not None:
            self.cam_intrinsic = np.array(camera_intrinsic)
        else:
            self.cam_intrinsic = np.array([100.6919557412736, 100.6919557412736, 160.0, 120.0, 320, 240])
        # Initialize occupancy grid in constructor
        self.get_occupancy_grid()
    
    def filter_point_cloud(self, pcd):
        """
        Filter noisy points from the point cloud using Open3D filtering methods.
        
        Args:
            pcd: Open3D point cloud object
            
        Returns:
            Filtered Open3D point cloud object
        """
        try:
            print(f"Original point cloud has {len(pcd.points)} points")
            
            # 1. Voxel downsampling to reduce noise and density
            voxel_size = 0.02  # 2cm voxel size
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
            print(f"After voxel downsampling: {len(pcd_downsampled.points)} points")
            
            # 2. Remove statistical outliers
            # This removes points that are too far from their neighbors
            pcd_cleaned, _ = pcd_downsampled.remove_statistical_outlier(
                nb_neighbors=20,  # Number of neighbors to analyze
                std_ratio=2.0     # Standard deviation ratio threshold
            )
            print(f"After statistical outlier removal: {len(pcd_cleaned.points)} points")
            
            # 3. Remove radius outliers (optional, can be commented out if too aggressive)
            # This removes points that have too few neighbors within a radius
            pcd_cleaned, _ = pcd_cleaned.remove_radius_outlier(
                nb_points=16,     # Minimum number of points within radius
                radius=0.05       # Radius to search for neighbors (5cm)
            )
            print(f"After radius outlier removal: {len(pcd_cleaned.points)} points")
            
            return pcd_cleaned
            
        except Exception as e:
            print(f"Error during point cloud filtering: {e}")
            print("Returning original point cloud without filtering")
            return pcd
    
    def filter_point_cloud_custom(self, pcd, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0, 
                                 nb_points=16, radius=0.05, use_radius_filter=True):
        """
        Filter noisy points from the point cloud with custom parameters.
        
        Args:
            pcd: Open3D point cloud object
            voxel_size: Size of voxels for downsampling (in meters)
            nb_neighbors: Number of neighbors for statistical outlier removal
            std_ratio: Standard deviation ratio threshold for statistical outlier removal
            nb_points: Minimum number of points within radius for radius outlier removal
            radius: Radius to search for neighbors (in meters)
            use_radius_filter: Whether to apply radius outlier removal
            
        Returns:
            Filtered Open3D point cloud object
        """
        try:
            print(f"Original point cloud has {len(pcd.points)} points")
            
            # 1. Voxel downsampling
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
            print(f"After voxel downsampling: {len(pcd_downsampled.points)} points")
            
            # 2. Remove statistical outliers
            pcd_cleaned, _ = pcd_downsampled.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            print(f"After statistical outlier removal: {len(pcd_cleaned.points)} points")
            
            # 3. Remove radius outliers (optional)
            if use_radius_filter:
                pcd_cleaned, _ = pcd_cleaned.remove_radius_outlier(
                    nb_points=nb_points,
                    radius=radius
                )
                print(f"After radius outlier removal: {len(pcd_cleaned.points)} points")
            
            return pcd_cleaned
            
        except Exception as e:
            print(f"Error during point cloud filtering: {e}")
            print("Returning original point cloud without filtering")
            return pcd
    
    def visualize_filtering_comparison(self, original_pcd):
        """
        Visualize the comparison between original and filtered point clouds.
        
        Args:
            original_pcd: Original Open3D point cloud object
        """
        try:
            # Create visualization window
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Point Cloud Filtering Comparison", width=1200, height=800)
            
            # Add original point cloud (white)
            original_pcd.paint_uniform_color([1, 1, 1])  # White
            vis.add_geometry(original_pcd)
            
            # Add filtered point cloud (red)
            filtered_pcd = self.pcd
            filtered_pcd.paint_uniform_color([1, 0, 0])  # Red
            vis.add_geometry(filtered_pcd)
            
            # Add coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            vis.add_geometry(coord_frame)
            
            # Set view options
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])  # Black background
            opt.point_size = 2.0
            
            print("Visualizing point cloud filtering comparison...")
            print("White points: Original point cloud")
            print("Red points: Filtered point cloud")
            print("Press 'Q' to close the visualization")
            
            # Run visualization
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            print(f"Error during visualization: {e}")
    
    def get_occupancy_grid(self):
        """Create occupancy grid from point cloud, filtering out ground points below ground_z"""
        # Filter out ground points
        non_ground_points = self.pcd_np[self.pcd_np[:, 2] >= self.ground_z]
        
        # Get min and max coordinates
        min_coords = np.min(non_ground_points, axis=0)
        max_coords = np.max(non_ground_points, axis=0)
        
        # Calculate grid dimensions
        width = int(np.ceil((max_coords[0] - min_coords[0]) / self.resolution))
        height = int(np.ceil((max_coords[1] - min_coords[1]) / self.resolution))
        
        # Initialize occupancy grid
        occupancy_grid = np.zeros((height, width))
        
        # Project points to grid
        for point in non_ground_points:
            x_idx = int((point[0] - min_coords[0]) / self.resolution)
            y_idx = int((point[1] - min_coords[1]) / self.resolution)
            
            if 0 <= x_idx < width and 0 <= y_idx < height:
                occupancy_grid[y_idx, x_idx] = 1
        
        self.occupancy_grid = occupancy_grid
        self.min_coords = min_coords
        self.max_coords = max_coords

    def check_boundary(self, se2_pose):
        """Check if the target is within the boundary"""
        x, y, theta = se2_pose
        if x < self.pcd_min_value[0] or x > self.pcd_max_value[0] or y < self.pcd_min_value[1] or y > self.pcd_max_value[1]:
            return False
        return True
    
    def check_collision(self, se2_pose):
        """Check if all grids intersected by the rectangle are empty"""
        x, y, theta = se2_pose
        # Get rectangle corners in world frame
        corners = np.array([
            [-self.length*0.83, -self.width/2],          # Bottom left
            [self.length*0.17, -self.width/2],           # Bottom right
            [self.length*0.17, self.width/2],            # Top right
            [-self.length*0.83, self.width/2]            # Top left
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
        min_x_idx = int((rect_min[0] - self.min_coords[0]) / self.resolution)
        min_y_idx = int((rect_min[1] - self.min_coords[1]) / self.resolution)
        max_x_idx = int((rect_max[0] - self.min_coords[0]) / self.resolution) + 1
        max_y_idx = int((rect_max[1] - self.min_coords[1]) / self.resolution) + 1
        
        # Ensure indices are within grid bounds
        min_x_idx = max(0, min_x_idx)
        min_y_idx = max(0, min_y_idx)
        max_x_idx = min(self.occupancy_grid.shape[1], max_x_idx)
        max_y_idx = min(self.occupancy_grid.shape[0], max_y_idx)
        
        # Check each grid cell in the bounding box
        for y_idx in range(min_y_idx, max_y_idx):
            for x_idx in range(min_x_idx, max_x_idx):
                if self.occupancy_grid[y_idx, x_idx] == 1:
                    # Get grid cell corners in world coordinates
                    grid_min_x = self.min_coords[0] + x_idx * self.resolution
                    grid_min_y = self.min_coords[1] + y_idx * self.resolution
                    grid_max_x = grid_min_x + self.resolution
                    grid_max_y = grid_min_y + self.resolution
                    
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
                    rect_min_x = -self.length*0.83
                    rect_max_x = self.length*0.17
                    rect_min_y = -self.width/2
                    rect_max_y = self.width/2
                    
                    # Check if any grid corner is inside rectangle
                    for corner in local_corners:
                        if (rect_min_x <= corner[0] <= rect_max_x and 
                            rect_min_y <= corner[1] <= rect_max_y):
                            return False
        
        return True
    
    def visualize_occupancy_and_rectangle(self, target_se2, object_pos=None):
        """Visualize occupancy grid and rectangle"""
        try:
            plt.figure(figsize=(10, 10))
            ax = plt.gca()
            
            # Plot occupancy grid
            plt.imshow(self.occupancy_grid, cmap='binary', origin='lower',
                      extent=[self.min_coords[0], self.max_coords[0], self.min_coords[1], self.max_coords[1]])
            
            # Draw sampling region
            sampling_rect = patches.Rectangle(
                (self.origin_se2[0] - self.x_half_range, self.origin_se2[1] - self.y_half_range),
                2 * self.x_half_range, 2 * self.y_half_range,
                linewidth=1, edgecolor='b', facecolor='none', linestyle='--', label='Random Region'
            )
            ax.add_patch(sampling_rect)
            
            # Draw origin point
            ax.plot(self.origin_se2[0], self.origin_se2[1], 'bo', markersize=8, label='Origin')
            
            # Draw rectangle with SE2 pose
            x, y, theta = target_se2
            
            # Create rectangle with origin at center of width edge
            rect = patches.Rectangle((-self.length*0.83, -self.width/2), 
                                   self.length, self.width, 
                                   linewidth=2, edgecolor='r', facecolor='none')
            
            # Create transformation
            t = transforms.Affine2D().rotate(theta).translate(x, y)
            rect.set_transform(t + ax.transData)
            
            # Add rectangle and reachability circle to plot
            ax.add_patch(rect)

            # Create reachability circle centered at object_pos
            if object_pos is not None:
                reachability_circle = patches.Circle(object_pos, self.arm_length, linewidth=2, edgecolor='b', facecolor='none')
                ax.add_patch(reachability_circle)
            
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
            
            # Add axis labels
            plt.xlabel('X (meters)')
            plt.ylabel('Y (meters)')
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            # Add title
            plt.title('Occupancy Grid Map with Rectangle')
            
            # Add legend
            plt.legend()
            
            # Show plot
            plt.show()
            plt.pause(0.5)
            plt.close()  # Explicitly close the figure
        except Exception as e:
            print(f"Visualization error: {e}")

    def visualize_pcl_with_camera_and_object(self, se2_pose, object_pos):
        """Visualize point cloud, camera, and object using Open3D"""
        try:
            camera_extrinsic = self.calculate_camera_extrinsic(se2_pose)
            se3_pose = self.calculate_target_se3(se2_pose)
            
            # Create visualization window
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            
            # Add point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.pcd_np[:,:3])
            pcd.colors = o3d.utility.Vector3dVector(self.pcd_color_np)
            vis.add_geometry(pcd)
            
            # Add object as sphere
            if object_pos.shape == (2,):
                object_pos = np.array([object_pos[0], object_pos[1], 1.0])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(object_pos[:3])
            sphere.paint_uniform_color([1, 0, 0])  # Red color
            vis.add_geometry(sphere)
            
            # # Add camera frustum
            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            camera_frame.transform(camera_extrinsic)
            vis.add_geometry(camera_frame)

            # Add target as sphere
            target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            target_frame.transform(se3_pose)
            vis.add_geometry(target_frame)
            
            # Add origin coordinate frame
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            vis.add_geometry(origin_frame)
            
            # Render and capture image
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            print(f"Visualization error: {e}")

    def calculate_target_se3(self, se2_pose):
        """Get the target SE3 pose"""
        x, y, theta = se2_pose
        se3_pose = np.array([
            [np.cos(theta), -np.sin(theta), 0, x],
            [np.sin(theta), np.cos(theta), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return se3_pose

    def calculate_camera_extrinsic(self, se2_pose):
        """Get the camera extrinsic matrix from the target SE2 pose"""
        se3_pose = self.calculate_target_se3(se2_pose)
        camera_extrinsic = se3_pose @ self.T_base_cam
        return camera_extrinsic

    def check_object_visibility(self, se2_pose, object_pos):
        """Check if the object is visible from the camera"""
        # convert target_se2 to camera_extrinsic
        camera_extrinsic = self.calculate_camera_extrinsic(se2_pose)
        fx, fy, cx, cy, w, h = self.cam_intrinsic
        intrinsic_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        if object_pos.shape == (3,):
            object_pos = np.concatenate([object_pos, np.array([1.0])])
        elif object_pos.shape == (2,): # use default height 1.0
            object_pos = np.concatenate([object_pos, np.array([1.0, 1.0])])
        else:
            raise ValueError("object_pos must be a 2D or 3D array")

        # convert object_pos to camera coordinates
        object_cam = np.linalg.inv(camera_extrinsic) @ object_pos
        object_cam = object_cam[:3] / object_cam[3]
        object_pix = intrinsic_matrix @ object_cam
        object_pix = object_pix[:2] / object_pix[2]

        if self.vis:
            global_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            global_frame_points = global_frame.sample_points_uniformly(number_of_points=1000)

            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            camera_frame.transform(camera_extrinsic)
            camera_frame_points = camera_frame.sample_points_uniformly(number_of_points=1000)

            base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
            base_frame.transform(self.calculate_target_se3(se2_pose))
            base_frame_points = base_frame.sample_points_uniformly(number_of_points=1000)
            
            object_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            object_point.translate(object_pos[:3])
            object_point.paint_uniform_color([1, 0, 0])
            object_point_points = object_point.sample_points_uniformly(number_of_points=1000)

            transformed_object_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            transformed_object_point.translate((np.linalg.inv(camera_extrinsic) @ object_pos)[:3])
            transformed_object_point.paint_uniform_color([0, 1, 0])
            transformed_object_point_points = transformed_object_point.sample_points_uniformly(number_of_points=1000)

            o3d.io.write_point_cloud("check_transform.ply", global_frame_points + camera_frame_points + base_frame_points + object_point_points + transformed_object_point_points)

        # check if the object is in front of the camera
        if object_cam[2] <= 0:
            return False
        
        # check if the object is within the image bounds
        if object_pix[0] < 0 or object_pix[0] > w or object_pix[1] < 0 or object_pix[1] > h:
            return False
        
        return True
    
    def check_reachability(self, se2_pose, object_pos):
        """Check if the object is reachable from the target"""
        # convert target_se2 to camera_extrinsic
        robot_xy = se2_pose[:2]
        object_xy = object_pos[:2]
        distance = np.linalg.norm(robot_xy - object_xy)
        if distance > self.arm_length:
            return False
        return True

    def get_random_target_se2(self):
        """Get a random target SE2 pose"""
        while True:
            random_x = np.random.uniform(-self.x_half_range, self.x_half_range)
            random_y = np.random.uniform(-self.y_half_range, self.y_half_range)
            random_theta = np.random.uniform(self.to_rad(-self.theta_half_range_deg), self.to_rad(self.theta_half_range_deg))
            # Calculate absolute position by adding to origin
            se2_delta = [random_x, random_y, random_theta]
            se2_pose = [
                self.origin_se2[0] + se2_delta[0],
                self.origin_se2[1] + se2_delta[1],
                self.origin_se2[2] + se2_delta[2]
            ]
            if self.check_collision(se2_pose):
                # print("collision, break", se2_pose)
                break
        
        if self.vis:
            try:
                self.visualize_occupancy_and_rectangle(se2_pose)
            except Exception as e:
                print(f"Visualization failed: {e}")
        
        return se2_delta
    
    def get_random_target_se2_with_reachability_check(self, object_pos):
        """Get a random target SE2 pose with reachability check"""
        while True:
            random_x = np.random.uniform(-self.x_half_range, self.x_half_range)
            random_y = np.random.uniform(-self.y_half_range, self.y_half_range)
            random_theta = np.random.uniform(self.to_rad(-self.theta_half_range_deg), self.to_rad(self.theta_half_range_deg))
            # Calculate absolute position by adding to origin
            se2_delta = [random_x, random_y, random_theta]
            se2_pose = [
                self.origin_se2[0] + se2_delta[0],
                self.origin_se2[1] + se2_delta[1],
                self.origin_se2[2] + se2_delta[2]
            ]
            if self.check_collision(se2_pose) and self.check_reachability(se2_pose, object_pos):
                break

        if self.vis:
            try:
                self.visualize_occupancy_and_rectangle(se2_pose, object_pos)
            except Exception as e:
                print(f"Visualization failed: {e}")

        return se2_delta
    
    def get_random_target_se2_with_visibility_check(self, object_pos):
        """Get a random target SE2 pose with visibility check"""
        while True:
            se2_delta = self.get_random_target_se2()
            se2_pose = [
                self.origin_se2[0] + se2_delta[0],
                self.origin_se2[1] + se2_delta[1],
                self.origin_se2[2] + se2_delta[2]
            ]
            camera_extrinsic = self.calculate_camera_extrinsic(se2_pose)
            if self.check_object_visibility(se2_pose, object_pos) and self.check_boundary(se2_pose):
                break

        if self.vis:
            try:
                self.visualize_pcl_with_camera_and_object(se2_pose, object_pos)
            except Exception as e:
                print(f"Visualization failed: {e}")
        
        return se2_delta, camera_extrinsic

    def get_random_target_se2_with_boundary_check(self, object_pos):
        """Get a random target SE2 pose with visibility check"""
        while True:
            se2_delta = self.get_random_target_se2()
            se2_pose = [
                self.origin_se2[0] + se2_delta[0],
                self.origin_se2[1] + se2_delta[1],
                self.origin_se2[2] + se2_delta[2]
            ]
            camera_extrinsic = self.calculate_camera_extrinsic(se2_pose)
            if self.check_boundary(se2_pose):
                break

        if self.vis:
            try:
                self.visualize_pcl_with_camera_and_object(se2_pose, object_pos)
            except Exception as e:
                print(f"Visualization failed: {e}")
        
        return se2_delta, camera_extrinsic
    
    def get_random_target_se2_with_visibility_check_without_boundary(self, object_pos):
        """Get a random target SE2 pose with visibility check"""
        while True:
            se2_delta = self.get_random_target_se2()
            se2_pose = [
                self.origin_se2[0] + se2_delta[0],
                self.origin_se2[1] + se2_delta[1],
                self.origin_se2[2] + se2_delta[2]
            ]
            camera_extrinsic = self.calculate_camera_extrinsic(se2_pose)
            if self.check_object_visibility(se2_pose, object_pos):
                break

        if self.vis:
            try:
                self.visualize_pcl_with_camera_and_object(se2_pose, object_pos)
            except Exception as e:
                print(f"Visualization failed: {e}")
        
        return se2_delta, camera_extrinsic
    
    def get_random_target_se2_with_visibility_reachability_check(self, object_pos):
        """Get a random target SE2 pose with visibility check"""
        while True:
            se2_delta = self.get_random_target_se2()
            se2_pose = [
                self.origin_se2[0] + se2_delta[0],
                self.origin_se2[1] + se2_delta[1],
                self.origin_se2[2] + se2_delta[2]
            ]
            camera_extrinsic = self.calculate_camera_extrinsic(se2_pose)
            if self.check_object_visibility(se2_pose, object_pos) and self.check_reachability(se2_pose, object_pos):
                break

        if self.vis:
            try:
                self.visualize_pcl_with_camera_and_object(se2_pose, object_pos)
            except Exception as e:
                print(f"Visualization failed: {e}")
        
        return se2_delta, camera_extrinsic