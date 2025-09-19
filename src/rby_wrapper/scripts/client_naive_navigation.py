#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist, Point, Quaternion, Vector3, Pose
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import numpy as np
from builtin_interfaces.msg import Duration

# map_or_odom
# FRAME_ID = "map"  
FRAME_ID = "odom"  

class NAIVE_PLANNER:
    def __init__(self):
        # Initialize current and target poses
        self.current_se2 = [0.0, 0.0, 0.0]  # x, y, theta
        self.target_se2 = [0.0, 0.0, 0.0]   # x, y, theta
        self.is_initialized = False
        
        # Define navigation states
        self.ROTATE_TO_TARGET = 0   # State 1: Rotate toward target point
        self.MOVE_TO_TARGET = 1     # State 2: Move straight to target point
        self.ROTATE_TO_TARGET_THETA = 2  # State 3: Rotate to target orientation
        
        # Initial state
        self.state = self.ROTATE_TO_TARGET
        
        # Set navigation parameters
        self.max_linear_speed = 0.2     # Maximum linear speed (m/s)
        self.max_angular_speed = 0.3    # Maximum angular speed (rad/s)
        self.position_tolerance = 0.03   # Position tolerance (m)
        self.angle_tolerance = 0.03     # Angle tolerance (rad)
        self.slowdown_distance = 0.5    # Distance to start slowing down (m)
        self.min_linear_speed = 0.05    # Minimum linear speed (m/s)
        
        # Store path points for visualization
        self.path_points = []
        self.planned_path_points = []   # For pre-visualization of planned path
        self.max_path_points = 500      # Maximum number of path points
        
        # Wheel offset in x direction based on URDF
        self.wheel_x_offset = 0.228     # Wheel position offset in x direction
        
        # Store both original and adjusted SE2 values
        self.original_current_se2 = [0.0, 0.0, 0.0]
        self.original_target_se2 = [0.0, 0.0, 0.0]
    
    def reset_state(self):
        """Reset navigation state"""
        self.state = self.ROTATE_TO_TARGET
        self.path_points = []  # Clear historical path
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def compute_shortest_rotation(self, current_theta, target_theta):
        """Calculate shortest rotation direction and angle difference"""
        # Ensure angles are in [-pi, pi] range
        current_theta = self.normalize_angle(current_theta)
        target_theta = self.normalize_angle(target_theta)
        
        # Calculate angle difference (shortest path)
        diff = self.normalize_angle(target_theta - current_theta)
        
        # diff is already the shortest path angle difference
        return diff
    
    def adjust_to_rotation_center(self, x, y, theta, x_offset):
        """
        Transform from base coordinates to rotation center coordinates
        For a differential drive robot, the rotation center is at the midpoint between the two wheels
        """
        # Calculate rotation center coordinates (which is forward by x_offset)
        rotation_center_x = x + x_offset * math.cos(theta)
        rotation_center_y = y + x_offset * math.sin(theta)
        
        return rotation_center_x, rotation_center_y, theta
    
    def adjust_from_rotation_center(self, x, y, theta, x_offset):
        """Transform from rotation center coordinates back to base coordinates"""
        # Calculate base coordinates (which is backward by x_offset)
        base_x = x - x_offset * math.cos(theta)
        base_y = y - x_offset * math.sin(theta)
        
        return base_x, base_y, theta
    
    def update_se2(self, current_se2, target_se2):
        """Update current and target poses, adjusting for wheel center offset"""
        # Store original values (base pose)
        self.original_current_se2 = current_se2.copy()
        self.original_target_se2 = target_se2.copy()
        
        # Transform from base coordinates to rotation center coordinates
        # The wheels are forward from the base by wheel_x_offset, so the rotation center is there
        adjusted_current_x, adjusted_current_y, current_theta = self.adjust_to_rotation_center(
            current_se2[0], current_se2[1], current_se2[2], self.wheel_x_offset
        )
        
        adjusted_target_x, adjusted_target_y, target_theta = self.adjust_to_rotation_center(
            target_se2[0], target_se2[1], target_se2[2], self.wheel_x_offset
        )
        
        # Update with adjusted values (now we're working with the rotation center)
        self.current_se2 = [adjusted_current_x, adjusted_current_y, current_theta]
        self.target_se2 = [adjusted_target_x, adjusted_target_y, target_theta]
        
        # Normalize angles
        self.current_se2[2] = self.normalize_angle(self.current_se2[2])
        self.target_se2[2] = self.normalize_angle(self.target_se2[2])
        self.is_initialized = True
        
        # Add current position to path points (use original for visualization)
        self.add_path_point(self.original_current_se2[0], self.original_current_se2[1])
        
        # Update planned path
        self.generate_planned_path()
    
    def add_path_point(self, x, y):
        """Add path point and limit maximum number"""
        self.path_points.append((x, y))
        if len(self.path_points) > self.max_path_points:
            self.path_points.pop(0)
    
    def generate_planned_path(self):
        """Generate planned path from current to target position"""
        if not self.is_initialized:
            return
        
        # Clear previous planned path
        self.planned_path_points = []
        
        # Calculate straight line path from current to target using original coordinates for visualization
        start_x, start_y = self.original_current_se2[0], self.original_current_se2[1]
        target_x, target_y = self.original_target_se2[0], self.original_target_se2[1]
        
        # Generate points along straight line
        distance = self.compute_distance_to_target()
        num_points = max(10, int(distance * 20))  # Number of points based on distance
        
        for i in range(num_points + 1):
            t = i / num_points
            x = start_x + t * (target_x - start_x)
            y = start_y + t * (target_y - start_y)
            self.planned_path_points.append((x, y))
    
    def compute_target_direction(self):
        """Calculate direction from current to target position"""
        dx = self.target_se2[0] - self.current_se2[0]
        dy = self.target_se2[1] - self.current_se2[1]
        return math.atan2(dy, dx)
    
    def compute_distance_to_target(self):
        """Calculate distance from current to target position"""
        dx = self.target_se2[0] - self.current_se2[0]
        dy = self.target_se2[1] - self.current_se2[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def get_control_command(self):
        """Calculate control command based on current state"""
        if not self.is_initialized:
            return 0.0, 0.0  # If not initialized, don't move
        
        linear_vel = 0.0
        angular_vel = 0.0
        
        if self.state == self.ROTATE_TO_TARGET:
            # Step 1: Rotate toward target point
            target_direction = self.compute_target_direction()
            angle_diff = self.compute_shortest_rotation(self.current_se2[2], target_direction)
            
            if abs(angle_diff) < self.angle_tolerance:
                # Rotation complete, move to next state
                self.state = self.MOVE_TO_TARGET
            else:
                # Calculate smooth angular velocity, slow down near target
                if abs(angle_diff) < self.max_angular_speed:
                    angular_vel = angle_diff  # Use angle difference directly for smooth deceleration
                else:
                    # Use maximum angular speed, but maintain rotation direction
                    angular_vel = self.max_angular_speed * (angle_diff / abs(angle_diff))
        
        elif self.state == self.MOVE_TO_TARGET:
            # Step 2: Move straight to target point
            distance = self.compute_distance_to_target()
            
            if distance < self.position_tolerance:
                # Reached target position, move to final rotation state
                self.state = self.ROTATE_TO_TARGET_THETA
            else:
                # May need slight direction adjustments
                target_direction = self.compute_target_direction()
                angle_diff = self.compute_shortest_rotation(self.current_se2[2], target_direction)
                
                # Adjust linear speed based on distance, slow down when approaching target
                if distance < self.slowdown_distance:
                    # Linear deceleration in slowdown zone
                    speed_factor = max(0.25, distance / self.slowdown_distance)
                    linear_vel = self.max_linear_speed * speed_factor
                    linear_vel = max(self.min_linear_speed, linear_vel)  # Ensure not below minimum speed
                else:
                    linear_vel = self.max_linear_speed
                
                # If off course, slow down and adjust direction
                if abs(angle_diff) > self.angle_tolerance:
                    linear_vel *= max(0.3, 1.0 - abs(angle_diff) / math.pi)
                    angular_vel = 0.5 * angle_diff  # Keep pointing toward target
        
        elif self.state == self.ROTATE_TO_TARGET_THETA:
            # Step 3: Rotate to target orientation
            angle_diff = self.compute_shortest_rotation(self.current_se2[2], self.target_se2[2])
            
            if abs(angle_diff) < self.angle_tolerance:
                # Completed all navigation steps
                linear_vel = 0.0
                angular_vel = 0.0
                print("Navigation completed!")
            else:
                # Calculate smooth angular velocity, slow down near target
                if abs(angle_diff) < self.max_angular_speed:
                    angular_vel = angle_diff  # Use angle difference directly for smooth deceleration
                else:
                    # Use maximum angular speed, but maintain rotation direction
                    angular_vel = self.max_angular_speed * (angle_diff / abs(angle_diff))
        
        return linear_vel, angular_vel

class NavigationController(Node):
    def __init__(self):
        super().__init__('navigation_controller')
        
        # Create planner
        self.planner = NAIVE_PLANNER()
        
        # Navigation enable flag
        self.is_nav_enabled = False
        self.prev_nav_enabled = False  # For detecting navigation state changes
        
        # Create subscribers
        self.se2_current_and_target_sub = self.create_subscription(
            Float32MultiArray, 
            '/robot/se2_current_and_target', 
            self.se2_callback, 
            10
        )
        
        self.nav_enable_sub = self.create_subscription(
            Bool,
            '/manager/is_nav',
            self.nav_enable_callback,
            10
        )
        
        # Create publisher for control commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist, 
            'cmd_vel', 
            10
        )
        
        # Create visualization publishers
        self.robot_marker_pub = self.create_publisher(
            Marker, 
            '/visualization/robot', 
            10
        )
        
        self.target_marker_pub = self.create_publisher(
            Marker, 
            '/visualization/target', 
            10
        )
        
        self.path_marker_pub = self.create_publisher(
            Marker, 
            '/visualization/path', 
            10
        )
        
        self.planned_path_pub = self.create_publisher(
            Marker, 
            '/visualization/planned_path', 
            10
        )
        
        self.target_direction_pub = self.create_publisher(
            Marker, 
            '/visualization/target_direction', 
            10
        )
        
        self.state_text_pub = self.create_publisher(
            Marker, 
            '/visualization/state_text', 
            10
        )
        
        # Create rotation visualization publisher
        self.rotation_visualization_pub = self.create_publisher(
            Marker,
            '/visualization/rotation_arc',
            10
        )
        
        # Create rotation center visualization
        self.rotation_center_pub = self.create_publisher(
            Marker,
            '/visualization/rotation_center',
            10
        )
        
        # Create timers
        self.timer = self.create_timer(0.1, self.control_loop)
        self.vis_timer = self.create_timer(0.2, self.visualization_loop)
        
        self.get_logger().info('Navigation controller initialized')
        self.get_logger().info(f'Using wheel offset: {self.planner.wheel_x_offset}m')
    
    def nav_enable_callback(self, msg):
        """Receive navigation enable signal"""
        new_nav_state = msg.data
        
        if new_nav_state:  # If the new state is active
            if not self.is_nav_enabled:  # If current state is inactive
                # Navigation changing from inactive to active
                self.planner.reset_state()
                self.get_logger().info('Navigation enabled - state reset')
        else:  # If the new state is inactive
            if self.is_nav_enabled:  # If current state is active
                # Navigation changing from active to inactive
                self.get_logger().info('Navigation disabled, stopping robot')
                stop_msg = Twist()
                self.cmd_vel_publisher.publish(stop_msg)
            else:
                # Already inactive, just ensure robot is stopped
                stop_msg = Twist()
                self.cmd_vel_publisher.publish(stop_msg)
        
        # Update current state
        self.is_nav_enabled = new_nav_state
            
    def se2_callback(self, msg):
        """Receive SE2 data and update planner"""
        if len(msg.data) == 6:  # Ensure correct data format
            current_se2 = list(msg.data[0:3])
            target_se2 = list(msg.data[3:6])
            self.planner.update_se2(current_se2, target_se2)
            
            # Log data (using original coordinates)
            self.get_logger().debug(f'Original Current SE2: [{current_se2[0]:.2f}, {current_se2[1]:.2f}, {current_se2[2]:.2f}]')
            self.get_logger().debug(f'Original Target SE2: [{target_se2[0]:.2f}, {target_se2[1]:.2f}, {target_se2[2]:.2f}]')
            self.get_logger().debug(f'Adjusted Current SE2: [{self.planner.current_se2[0]:.2f}, {self.planner.current_se2[1]:.2f}, {self.planner.current_se2[2]:.2f}]')
            self.get_logger().debug(f'Adjusted Target SE2: [{self.planner.target_se2[0]:.2f}, {self.planner.target_se2[1]:.2f}, {self.planner.target_se2[2]:.2f}]')
        else:
            self.get_logger().error(f'Invalid SE2 data format. Expected 6 values but got {len(msg.data)}')
    
    def control_loop(self):
        """Main control loop, periodically update control commands"""
        # If navigation not enabled or planner not initialized, don't publish commands
        if not self.is_nav_enabled or not self.planner.is_initialized:
            return
        
        # Get control commands
        linear_vel, angular_vel = self.planner.get_control_command()
        
        # Create and publish Twist message
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self.cmd_vel_publisher.publish(twist_msg)
        
        # Log navigation state
        state_names = ["ROTATE_TO_TARGET", "MOVE_TO_TARGET", "ROTATE_TO_TARGET_THETA"]
        self.get_logger().debug(f'State: {state_names[self.planner.state]}, linear: {linear_vel:.2f}, angular: {angular_vel:.2f}')
    
    def create_arrow_marker(self, x, y, theta, id, r, g, b, scale=0.3, namespace="arrows"):
        """Create an arrow marker to represent position and orientation"""
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Set arrow start and end points
        start = Point(x=x, y=y, z=0.05)
        end = Point(x=x + scale * math.cos(theta), 
                    y=y + scale * math.sin(theta), 
                    z=0.05)
        
        marker.points = [start, end]
        
        # Set arrow size
        marker.scale.x = 0.05  # Arrow shaft diameter
        marker.scale.y = 0.1   # Arrow head diameter
        marker.scale.z = 0.0   # Not used
        
        # Set color
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        
        # Set duration (0 means permanent)
        marker.lifetime = Duration(sec=0, nanosec=0)
        
        return marker
    
    def create_path_marker(self, points, id=0, r=0.0, g=1.0, b=0.0, alpha=0.8, z_height=0.01):
        """Create a line marker to represent path"""
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "path"
        marker.id = id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Add path points
        for p in points:
            pt = Point(x=p[0], y=p[1], z=z_height)
            marker.points.append(pt)
        
        # Set line thickness
        marker.scale.x = 0.03  # Line width
        
        # Set color
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = alpha
        
        # Set duration (0 means permanent)
        marker.lifetime = Duration(sec=0, nanosec=0)
        
        return marker
    
    def create_rotation_arc_marker(self, x, y, start_angle, end_angle, id=0, r=1.0, g=0.7, b=0.2):
        """Create a rotation arc marker to show rotation direction"""
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rotation_arc"
        marker.id = id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Calculate number of arc points
        angle_diff = self.planner.normalize_angle(end_angle - start_angle)
        num_points = max(10, int(abs(angle_diff) * 20 / math.pi))  # Points based on angle difference
        
        # Radius slightly larger than robot
        radius = 0.4
        
        # Create arc points
        for i in range(num_points + 1):
            t = i / num_points
            # Determine rotation direction based on angle difference sign
            if angle_diff >= 0:  # Counter-clockwise
                angle = start_angle + t * angle_diff
            else:  # Clockwise
                angle = start_angle + t * angle_diff
                
            # Calculate point coordinates
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            
            marker.points.append(Point(x=px, y=py, z=0.05))
        
        # Set line thickness
        marker.scale.x = 0.04  # Line width
        
        # Set color
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 0.8
        
        # Set duration (0 means permanent)
        marker.lifetime = Duration(sec=0, nanosec=0)
        
        return marker
    
    def create_text_marker(self, x, y, text, id=0, r=1.0, g=1.0, b=1.0):
        """Create a text marker"""
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "text"
        marker.id = id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.5  # Text height
        
        marker.pose.orientation.w = 1.0
        
        marker.scale.z = 0.2  # Text size
        
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        
        marker.text = text
        
        marker.lifetime = Duration(sec=0, nanosec=0)
        
        return marker
    
    def create_sphere_marker(self, x, y, r, g, b, scale=0.1, id=0, namespace="sphere"):
        """Create a sphere marker"""
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.05
        
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        
        marker.lifetime = Duration(sec=0, nanosec=0)
        
        return marker
    
    def visualization_loop(self):
        """Visualization loop, periodically update RViz visualization content"""
        if not self.planner.is_initialized:
            return
        
        # Use original coordinates for visualization
        original_current_x = self.planner.original_current_se2[0]
        original_current_y = self.planner.original_current_se2[1]
        original_current_theta = self.planner.original_current_se2[2]
        
        original_target_x = self.planner.original_target_se2[0]
        original_target_y = self.planner.original_target_se2[1]
        original_target_theta = self.planner.original_target_se2[2]
        
        # 1. Robot current position and orientation visualization
        robot_marker = self.create_arrow_marker(
            original_current_x, 
            original_current_y,
            original_current_theta,
            0, 0.0, 0.0, 1.0,  # Blue
            scale=0.5,
            namespace="robot"
        )
        self.robot_marker_pub.publish(robot_marker)
        
        # 2. Target position and orientation visualization
        target_marker = self.create_arrow_marker(
            original_target_x, 
            original_target_y,
            original_target_theta,
            0, 1.0, 0.0, 0.0,  # Red
            scale=0.5,
            namespace="target"
        )
        self.target_marker_pub.publish(target_marker)
        
        # 3. Historical path visualization (only when navigation is active)
        if len(self.planner.path_points) > 1 and self.is_nav_enabled:
            path_marker = self.create_path_marker(self.planner.path_points, r=0.0, g=0.5, b=1.0)
            self.path_marker_pub.publish(path_marker)
        
        # 4. Planned path visualization (regardless of navigation activation)
        if len(self.planner.planned_path_points) > 1:
            # Planned path shown as dashed line (through Alpha channel and height)
            planned_path_marker = self.create_path_marker(
                self.planner.planned_path_points, 
                id=1, 
                r=0.0, g=1.0, b=0.0,  # Green
                alpha=0.6,
                z_height=0.02  # Slightly higher
            )
            self.planned_path_pub.publish(planned_path_marker)
        
        # 5. Target direction visualization (regardless of navigation activation)
        # For visualization, we need to transform the direction back to the original frame
        target_direction = self.planner.compute_target_direction()
        
        # Create direction marker from original position
        direction_marker = self.create_arrow_marker(
            original_current_x, 
            original_current_y,
            target_direction,  # Direction is the same in both frames
            0, 0.0, 1.0, 0.0,  # Green
            scale=0.3,
            namespace="target_direction"
        )
        self.target_direction_pub.publish(direction_marker)
        
        # 6. Rotation arc visualization - using adjusted coordinates for rotation center
        if self.planner.state == self.planner.ROTATE_TO_TARGET:
            # Rotation to target position direction
            current_angle = self.planner.current_se2[2]
            target_angle = self.planner.compute_target_direction()
            rotation_marker = self.create_rotation_arc_marker(
                self.planner.current_se2[0],  # Use adjusted center for rotation
                self.planner.current_se2[1],
                current_angle,
                target_angle,
                0, 1.0, 0.7, 0.2  # Orange
            )
            self.rotation_visualization_pub.publish(rotation_marker)
        elif self.planner.state == self.planner.ROTATE_TO_TARGET_THETA:
            # Rotation to target orientation
            current_angle = self.planner.current_se2[2]
            target_angle = self.planner.target_se2[2]
            rotation_marker = self.create_rotation_arc_marker(
                self.planner.current_se2[0],  # Use adjusted center for rotation
                self.planner.current_se2[1],
                current_angle,
                target_angle,
                0, 1.0, 0.3, 0.3  # Purple-red
            )
            self.rotation_visualization_pub.publish(rotation_marker)
        
        # 7. State text visualization
        state_names = ["Rotating to Target", "Moving to Target", "Rotating to Target Orientation"]
        if self.is_nav_enabled:
            state_text = state_names[self.planner.state]
            # Show distance information
            distance = self.planner.compute_distance_to_target()
            state_text += f" (Distance: {distance:.2f}m)"
            
            # If in rotation state, add angle information
            if self.planner.state == self.planner.ROTATE_TO_TARGET:
                angle_diff = self.planner.compute_shortest_rotation(
                    self.planner.current_se2[2], 
                    self.planner.compute_target_direction()
                )
                state_text += f" (Angle diff: {abs(angle_diff)*180/math.pi:.1f}°)"
            elif self.planner.state == self.planner.ROTATE_TO_TARGET_THETA:
                angle_diff = self.planner.compute_shortest_rotation(
                    self.planner.current_se2[2], 
                    self.planner.target_se2[2]
                )
                state_text += f" (Angle diff: {abs(angle_diff)*180/math.pi:.1f}°)"
                
        else:
            state_text = "Navigation Disabled (Click to Start)"
        
        text_marker = self.create_text_marker(
            original_current_x, 
            original_current_y + 0.5,  # Display above robot
            state_text
        )
        self.state_text_pub.publish(text_marker)
        
        # 8. Visualize rotation center
        if self.planner.is_initialized:
            # Create a small sphere at the adjusted position (actual rotation center)
            rotation_center_marker = self.create_sphere_marker(
                self.planner.current_se2[0],
                self.planner.current_se2[1],
                1.0, 0.0, 1.0,  # Magenta
                scale=0.08,
                id=0,
                namespace="rotation_center"
            )
            self.rotation_center_pub.publish(rotation_center_marker)

def main(args=None):
    rclpy.init(args=args)
    navigation_controller = NavigationController()
    
    try:
        rclpy.spin(navigation_controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot
        stop_msg = Twist()
        navigation_controller.cmd_vel_publisher.publish(stop_msg)
        
        # Clean up resources
        navigation_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()