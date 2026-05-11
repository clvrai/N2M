"""Reachability predictor - finds reachable poses using IK.

TODO: To be implemented by user.

The basic idea is to iteratively sample poses and check:
1. Collision-free
2. IK-reachable (robot arm can reach target object from this base pose)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from benchmark.predictor.base import BasePredictor
from benchmark.utils.sampling_utils import sample_collision_free_pose
from benchmark.utils.collision_utils import CollisionChecker
from benchmark.utils.obs_utils import SE2_to_SE3
import pinocchio as pin
from tqdm import tqdm
import time


class ReachabilityPredictor(BasePredictor):
    """Reachability-based predictor using IK checking.
    
    TODO: User implementation required.
    
    Algorithm outline:
    1. Sample a candidate base pose in the task region
    2. Check collision-free
    3. Use IK solver to check if robot arm can reach target object
    4. If both checks pass, return this pose (done=True)
    5. Otherwise, continue sampling (done=False) until max tries
    
    This is an iterative predictor - may need multiple predict() calls.
    """
    
    def __init__(self, hydra_cfg, json_config, env, unwrapped_env):
        """Initialize reachability predictor.
        
        Args:
            hydra_cfg: Hydra config
            json_config: Robomimic/Robocasa config
            env: Environment instance (step)
            unwrapped_env: Unwrapped environment instance (forward)
        """
        super().__init__()  # BasePredictor.__init__() takes no arguments
        
        self.hydra_cfg = hydra_cfg
        self.json_config = json_config
        self.env = env
        self.unwrapped_env = unwrapped_env
        
        self.load_robot_model()
        
        # Define target surfaces (world frame)
        self.target_surface_1 = {
            'left_up': [3.5, -0.9, 0.9],
            'right_up': [4.2, -0.9, 0.9],
            'left_down': [3.5, -0.9, 0.7],
            'right_down': [4.2, -0.9, 0.7]
        }
        self.target_surface_2 = {
            'left_up': [3.5, -0.6, 0.7],
            'right_up': [4.2, -0.6, 0.7],
            'left_down': [3.5, -0.6, 0.9],
            'right_down': [4.2, -0.6, 0.9]
        }
        # base frame angle range
        self.target_angle_range = {
            'roll': [-np.pi/3, np.pi/3],
            # 'roll': [-np.pi, np.pi],
            'pitch': [-np.pi/3, np.pi/3],
            'yaw': [-np.pi/3, np.pi/3],
        }
        
        self.if_vis_sample = False
        
        # Pre-sample points on surfaces (only once)
        self.surface1_sampled = self.sample_points_on_surface(self.target_surface_1, num_samples=4)
        self.surface2_sampled = self.sample_points_on_surface(self.target_surface_2, num_samples=4)
        
        # Pre-generate target orientations (only once)
        self.target_orientations = self.generate_target_orientations()
        
        # Calculate surface centers for distance filtering
        self.s1_center = np.mean([self.target_surface_1[k] for k in self.target_surface_1.keys()], axis=0)
        self.s2_center = np.mean([self.target_surface_2[k] for k in self.target_surface_2.keys()], axis=0)
        
        
        # Storage for visualization
        self.visualization_data = []
    
    def predict(self, se2_initial, se2_randomized, collision_checker: CollisionChecker, episode_id=None):
        """Predict reachable base pose."""
        
        # Distance threshold for pre-filtering
        max_reach_distance = 1.2  # meters
        
        target_valid_num = 1  # Number of valid poses to find
        valid_count = 0  # Number of valid poses found so far
        iteration_count = 0  # Total iterations
        skipped_count = 0
        best_candidate = None
        
        # Create progress bar
        pbar = tqdm(total=target_valid_num, desc=f"Finding reachable poses (Episode {episode_id})", 
                   unit="poses", leave=True)
        
        time_start = time.time()
        while valid_count < target_valid_num:
            iteration_count += 1
            # print(f"\n{'─'*80}")
            # print(f"[ITERATION {iteration_count}] (Valid: {valid_count}/{target_valid_num})")
            # print(f"{'─'*80}")
            
            IK_randomization = {
                'x': [-1.2, 1.2],
                'y': [-0.8, 0.1],
                'theta': [-7*np.pi/6, 7*np.pi/6]
            }
            sample_center = [(self.s1_center[0]+self.s2_center[0])/2, (self.s1_center[1]+self.s2_center[1])/2, se2_initial[2]]
            object_pos = [(self.s1_center[0]+self.s2_center[0])/2, (self.s1_center[1]+self.s2_center[1])/2, self.s1_center[2]]

            # step1: resample a collision-free pose
            se2_candidate = sample_collision_free_pose(
                collision_checker,
                IK_randomization,
                se2_initial=sample_center,
                max_tries=100,
                visualize=self.if_vis_sample,
                save_path=f"debug1/episode_{episode_id}_{iteration_count}_sampling.png",
                object_pos=np.array(object_pos),
                check_visibility=False,
                check_boundary=False
            )

            if se2_candidate is None:
                continue

            # Check visibility (robot orientation should face the target within 45 degrees)
            robot_angle = np.degrees(se2_candidate[2])
            vect = np.array(object_pos[:2]) - np.array(se2_candidate[:2])
            target_angle = np.degrees(np.arctan2(vect[1], vect[0]))
            
            angle_diff = np.abs(robot_angle - target_angle)
            angle_diff = min(angle_diff, 360 - angle_diff)
            
            if angle_diff > 60:
                skipped_count += 1
                continue
            
            # step2: Pre-filter by distance to surface centers
            base_pos_2d = se2_candidate[:2]  # [x, y]
            
            # Calculate distances from base to surface centers (2D distance)
            dist_to_s1 = np.linalg.norm(base_pos_2d - self.s1_center[:2])
            dist_to_s2 = np.linalg.norm(base_pos_2d - self.s2_center[:2])
            
            # Skip if one of the surfaces are too far
            if dist_to_s1 > max_reach_distance or dist_to_s2 > max_reach_distance:
                skipped_count += 1
                continue
            # step3: Transform surfaces to panda_base frame
            SE3_pandaBase = SE2_to_SE3(se2_candidate, 0.7)
            
            # Get transformation matrix (world to panda_base)
            T_world_to_base = np.linalg.inv(SE3_pandaBase)
            
            # Transform pre-sampled points to base frame
            surface1_points_base = self.transform_points_to_base(self.surface1_sampled, T_world_to_base)
            surface2_points_base = self.transform_points_to_base(self.surface2_sampled, T_world_to_base)
            
            # step4: check IK reachability
            reach1, q_solution1 = self.check_points_reachability(surface1_points_base, SE3_pandaBase, surface_name="Surface 1")
            
            # If surface 1 is not reachable, skip surface 2 and continue to next candidate
            if not reach1:
                continue
            
            reach2, q_solution2 = self.check_points_reachability(surface2_points_base, SE3_pandaBase, surface_name="Surface 2")
            
            if reach1 and reach2:
                # Store visualization data
                self.visualization_data.append({
                    'se3_base': SE3_pandaBase,
                    'se2_base': se2_candidate,
                    'q_surface1': q_solution1,
                    'q_surface2': q_solution2,
                    'surface1_base': surface1_points_base,
                    'surface2_base': surface2_points_base,
                    'surface1_world': self.target_surface_1,
                    'surface2_world': self.target_surface_2
                })
                
                valid_count += 1
                best_candidate = se2_candidate
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'iterations': iteration_count,
                    'skipped': skipped_count,
                    'success_rate': f"{valid_count/iteration_count:.1%}"
                })
                
                # Save base frame points as image
                if self.if_vis_sample:
                    self.save_base_frame_visualization(surface1_points_base, surface2_points_base, iteration_count, q_solution1, q_solution2, SE3_pandaBase)
            else:
                continue
        time_end = time.time()
        
        # Close progress bar
        pbar.close()
        
        if len(self.visualization_data) > 0 and episode_id is not None:
            self.visualize_reachability(episode_id)
        
        # Return result
        if best_candidate is not None:
            se2_predicted = best_candidate
            extra_info = {
                'iterations': iteration_count,
                'pure_time': time_end - time_start
            }
        else:
            # Fallback to randomized pose if no reachable pose found
            se2_predicted = se2_randomized
            extra_info = {
                'iterations': iteration_count,
                'pure_time': time_end - time_start
            }
        
        result = {
            'is_ego': False,
            'se2_predicted': se2_predicted,
            'extra_info': extra_info
        }
        return result
    
    def load_robot_model(self):
        from robot_descriptions.loaders.yourdfpy import load_robot_description
        import tempfile
        import os
        
        # Load URDF
        urdf = load_robot_description('panda_description', load_meshes=False)
        urdf_bytes = urdf.write_xml_string()

        # Save to temp file
        temp_urdf = tempfile.NamedTemporaryFile(mode='wb', suffix='.urdf', delete=False)
        temp_urdf.write(urdf_bytes)
        temp_urdf.close()
        
        # load model
        self.model = pin.buildModelFromUrdf(temp_urdf.name)
        self.data = self.model.createData()
        print("Model: ", self.model)
        ''' Model:  Nb joints = 10 (nq=9,nv=9)
            Joint 0 universe: parent=0
            Joint 1 panda_joint1: parent=0
            Joint 2 panda_joint2: parent=1
            Joint 3 panda_joint3: parent=2
            Joint 4 panda_joint4: parent=3
            Joint 5 panda_joint5: parent=4
            Joint 6 panda_joint6: parent=5
            Joint 7 panda_joint7: parent=6
            Joint 8 panda_finger_joint1: parent=7
            Joint 9 panda_finger_joint2: parent=7
        '''
        # remove temp file
        os.unlink(temp_urdf.name)
    
    def transform_points_to_base(self, points_world, T_world_to_base):
        """Transform points from world frame to base frame."""
        points_base = []
        for point_world in points_world:
            point_world_h = np.array([point_world[0], point_world[1], point_world[2], 1.0])
            point_base_h = T_world_to_base @ point_world_h
            points_base.append(point_base_h[:3])
        return points_base
    
    def sample_points_on_surface(self, surface_dict, num_samples=5):
        """Sample points uniformly on a quadrilateral surface."""
        # Get corners
        lu = np.array(surface_dict['left_up'])
        ru = np.array(surface_dict['right_up'])
        ld = np.array(surface_dict['left_down'])
        rd = np.array(surface_dict['right_down'])
        
        points = []
        for i in range(num_samples):
            for j in range(num_samples):
                u = i / (num_samples - 1) if num_samples > 1 else 0.5
                v = j / (num_samples - 1) if num_samples > 1 else 0.5
                
                # Bilinear interpolation
                point = (1-u)*(1-v)*ld + (1-u)*v*lu + u*(1-v)*rd + u*v*ru
                points.append(point)
        
        return points
    
    def generate_target_orientations(self):
        """Generate multiple target orientations for IK in robot base frame using configured ranges."""
        import numpy as np
        
        orientations = []
        
        # Use the configured angle ranges with interpolation
        roll_range = self.target_angle_range['roll']
        pitch_range = self.target_angle_range['pitch'] 
        yaw_range = self.target_angle_range['yaw']
        
        # Generate discrete angles with π/4 intervals as suggested
        interval = np.pi / 6  # 45 degrees
        
        # Generate angle arrays
        roll_angles = np.arange(roll_range[0], roll_range[1] + interval, interval)
        pitch_angles = np.arange(pitch_range[0], pitch_range[1] + interval, interval)
        yaw_angles = np.arange(yaw_range[0], yaw_range[1] + interval, interval)
        
        # Generate rotation matrices (ZYX Euler angles)
        for yaw in yaw_angles:
            for pitch in pitch_angles:
                for roll in roll_angles:
                    # Create rotation matrix from Euler angles (ZYX convention)
                    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                   [np.sin(yaw), np.cos(yaw), 0],
                                   [0, 0, 1]])
                    
                    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                                   [0, 1, 0],
                                   [-np.sin(pitch), 0, np.cos(pitch)]])
                    
                    R_x = np.array([[1, 0, 0],
                                   [0, np.cos(roll), -np.sin(roll)],
                                   [0, np.sin(roll), np.cos(roll)]])
                    
                    R = R_z @ R_y @ R_x
                    orientations.append(R)
        
        return orientations
    
    def check_points_reachability(self, target_points, robot_se3, surface_name="Surface"):
        """Check if panda_joint7 can reach any point in the list."""
        
        # Get joint7 frame id
        joint7_id = self.model.getFrameId('panda_joint7')
        
        # Use pre-computed orientations
        for idx, target_pos in enumerate(target_points):
            if np.linalg.norm(target_pos) > 0.9:
                continue
            for ori_idx, R_base in enumerate(self.target_orientations):
                # Create target SE3 in base frame
                target_se3 = pin.SE3(R_base, target_pos)
                q_sol = self.solve_ik(target_se3, joint7_id)
                
                if q_sol is not None:
                    return True, q_sol
        
        return False, None
    
    def solve_ik(self, target_se3, frame_id, max_iter=100, eps=1e-2):
        """Solve IK using Pinocchio's iterative solver - 6D (position + rotation)."""
        # Simple single initial configuration
        q = pin.neutral(self.model)
        
        # IK solver parameters
        DT = 1e-1
        damp = 1e-6
        
        
        for i in range(max_iter):
            # Compute forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            # Get current frame pose
            current_se3 = self.data.oMf[frame_id]
            
            # Compute 6D error (position + rotation)
            error = pin.log(current_se3.inverse() * target_se3).vector
            error_norm = np.linalg.norm(error)
            
            
            # Check convergence
            if error_norm < eps:
                # Check joint limits
                if self.check_joint_limits(q):
                    return q
                else:
                    return None
            
            # Check if target is reachable (rough workspace check)
            target_distance = np.linalg.norm(target_se3.translation)
            if target_distance > 1.0:
                return None
            
            # Compute Jacobian in world frame (6x7 matrix)
            J = pin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.ReferenceFrame.WORLD)
            
            # Damped least squares for full 6D
            try:
                J_pinv = J.T @ np.linalg.inv(J @ J.T + damp * np.eye(6))
            except np.linalg.LinAlgError:
                # If matrix is singular, use larger damping
                J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6))
            
            # Update configuration
            dq = J_pinv @ error
            q = pin.integrate(self.model, q, dq * DT)
        
        return None  # Failed to converge
    
    def check_joint_limits(self, q):
        """Check if configuration is within joint limits."""
        return np.all(q >= self.model.lowerPositionLimit) and np.all(q <= self.model.upperPositionLimit)
    
    def visualize_reachability(self, episode_id):
        """Visualize target surfaces and robot arm configurations in 3D."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all collected samples
        max_samples = len(self.visualization_data)
        
        for idx, data in enumerate(self.visualization_data[:max_samples]):
            se3_base = data['se3_base']
            q1 = data['q_surface1']
            q2 = data['q_surface2']
            surface1 = data['surface1_world']
            surface2 = data['surface2_world']
            
            # Plot base position
            base_pos = se3_base[:3, 3]
            ax.scatter(*base_pos, c='red', marker='o', s=100, label='Robot Bases' if idx == 0 else '')
            
            # Plot robot orientation arrow
            # Extract rotation matrix and get forward direction (x-axis)
            R = se3_base[:3, :3]
            forward_dir = R[:, 0]  # x-axis of robot base frame
            arrow_length = 0.3  # 30cm arrow
            
            # Draw arrow showing robot orientation
            ax.quiver(base_pos[0], base_pos[1], base_pos[2],
                     forward_dir[0], forward_dir[1], forward_dir[2],
                     length=arrow_length, color='red', arrow_length_ratio=0.2,
                     linewidth=3, alpha=0.8)
            
            # Plot arm configuration for surface 1
            if q1 is not None:
                self.plot_arm_configuration(ax, q1, se3_base, color='blue', alpha=0.3)
            
            # Plot arm configuration for surface 2
            if q2 is not None:
                self.plot_arm_configuration(ax, q2, se3_base, color='green', alpha=0.3)
        
        # Plot target surfaces (only once)
        if len(self.visualization_data) > 0:
            surface1 = self.visualization_data[0]['surface1_world']
            surface2 = self.visualization_data[0]['surface2_world']
            self.plot_surface(ax, surface1, color='cyan', alpha=0.5, label='Surface 1 (open)')
            self.plot_surface(ax, surface2, color='yellow', alpha=0.5, label='Surface 2 (close)')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f'Reachability Visualization - Episode {episode_id}')
        
        # Save figure
        import os
        os.makedirs('debug', exist_ok=True)
        plt.savefig(f'debug/episode_{episode_id}_reachability.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_base_frame_visualization(self, surface1_points_base, surface2_points_base, iter_count, q1, q2, se3_base):
        """Save base frame points as 3D scatter plot with arm configurations."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import os
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract x, y coordinates for surface 1
        s1_x = [pt[0] for pt in surface1_points_base]
        s1_y = [pt[1] for pt in surface1_points_base]
        s1_z = [pt[2] for pt in surface1_points_base]
        
        # Extract x, y coordinates for surface 2
        s2_x = [pt[0] for pt in surface2_points_base]
        s2_y = [pt[1] for pt in surface2_points_base]
        s2_z = [pt[2] for pt in surface2_points_base]
        
        # Plot surface 1 points (3D)
        ax.scatter(s1_x, s1_y, s1_z, c='blue', s=50, alpha=0.7, 
                  label=f'Surface 1 ({len(surface1_points_base)} points)', marker='o')
        
        # Plot surface 2 points (3D)
        ax.scatter(s2_x, s2_y, s2_z, c='red', s=50, alpha=0.7,
                  label=f'Surface 2 ({len(surface2_points_base)} points)', marker='s')
        
        # Add robot base at origin
        ax.scatter(0, 0, 0, c='black', s=200, marker='*', label='Robot Base', zorder=10)
        
        # Add base coordinate frame arrows (3D)
        arrow_length = 0.3  # 30cm arrows
        # X-axis (forward direction) - red arrow
        ax.quiver(0, 0, 0, arrow_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=3, label='X-axis (forward)')
        # Y-axis (left direction) - green arrow  
        ax.quiver(0, 0, 0, 0, arrow_length, 0, color='green', arrow_length_ratio=0.1, linewidth=3, label='Y-axis (left)')
        # Z-axis (up direction) - blue arrow
        ax.quiver(0, 0, 0, 0, 0, arrow_length, color='blue', arrow_length_ratio=0.1, linewidth=3, label='Z-axis (up)')
        
        # Plot arm configurations
        if q1 is not None:
            # Create identity SE3 for base frame (since we're already in base frame)
            base_se3_identity = np.eye(4)
            self.plot_arm_configuration(ax, q1, base_se3_identity, color='cyan', alpha=0.8)
            
        if q2 is not None:
            base_se3_identity = np.eye(4)
            self.plot_arm_configuration(ax, q2, base_se3_identity, color='magenta', alpha=0.8)
        
        # Set labels and title
        ax.set_xlabel('X (m) - Base Frame')
        ax.set_ylabel('Y (m) - Base Frame')
        ax.set_zlabel('Z (m) - Base Frame')
        ax.set_title(f'Base Frame Target Points & Arm Configs - Iteration #{iter_count}')
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # Set equal aspect ratio for 3D
        max_range = 1.0  # 1 meter range
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range*2])
        
        # Save figure
        os.makedirs('debug', exist_ok=True)
        save_path = f'debug/valid_{iter_count}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_surface(self, ax, surface_dict, color='blue', alpha=0.5, label=None):
        """Plot a quadrilateral surface."""
        # Get corners in order: left_down, right_down, right_up, left_up
        vertices = [
            surface_dict['left_down'],
            surface_dict['right_down'],
            surface_dict['right_up'],
            surface_dict['left_up']
        ]
        
        # Create polygon
        poly = Poly3DCollection([vertices], alpha=alpha, facecolor=color, edgecolor='black')
        ax.add_collection3d(poly)
        
        # Plot vertices
        for v in vertices:
            ax.scatter(*v, c=color, s=20)
    
    def plot_arm_configuration(self, ax, q, base_se3, color='blue', alpha=0.5):
        """Plot robot arm in given configuration."""
        # Compute forward kinematics
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get joint positions and transform to world frame
        joint_positions = []
        
        # Add base position first
        base_pos = base_se3[:3, 3]
        joint_positions.append(base_pos)
        
        for i in range(1, 8):  # panda_joint1 to panda_joint7
            joint_name = f'panda_joint{i}'
            if self.model.existFrame(joint_name):
                frame_id = self.model.getFrameId(joint_name)
                # Get position in robot base frame
                pos_in_base = self.data.oMf[frame_id].translation
                # Transform to world frame: R * p_base + t
                pos_world = base_se3[:3, :3] @ pos_in_base + base_se3[:3, 3]
                joint_positions.append(pos_world)
        
        # Plot links
        if len(joint_positions) > 1:
            joint_positions = np.array(joint_positions)
            ax.plot(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2],
                   color=color, alpha=alpha, linewidth=2, marker='o', markersize=4)
            ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2],
                      c=color, alpha=alpha, s=30)

    def reset(self):
        """Reset internal state."""
        self.sample_count = 0
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint (not needed for reachability predictor)."""
        # No model to load - this is a heuristic method
        pass
    
    @property
    def name(self) -> str:
        """Return predictor name."""
        return "reachability"
