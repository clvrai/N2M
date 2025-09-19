#!/usr/bin/env python3
# Flow:
# 1. [function-1] Press arrow key to teleop base movement
# 2. [function-2] Press 1/2/3 to switch predefined Z
# 3. [function-3] Press r/R to reset the robot pose and clear point cloud
# 4. [function-4] Press c/C to collect and Stitch Point Clouds
# 5. [function-5] Press p/P to call the policy
# 6. [function-6] Press t/F to save the point cloud and other things
# 7. [function-7] Press q to quit
# 8. [function-8] Press h to toggle navigation mode

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import Empty, String, Float32, Bool
from rby_wrapper.msg import RobotControl, RobotState
import threading
from select import select
import numpy as np
import tkinter as tk
from tkinter import font
from util_pose import INIT_POSE
import signal
import sys
import queue
import time
import argparse
import tty
import termios

# Base ArrowKeyTeleop class (copied from utils/base_controller.py)
class ArrowKeyTeleop(Node):
    def __init__(self, norender=False):
        super().__init__('arrow_key_teleop')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info('Starting teleop control window. Click on window to control.')
        
        # Speed settings
        self.linear_speed_x = 0.0  # m/s
        self.angular_speed_z = 0.0  # rad/s
        self.speed_increment = 0.1
        self.max_speed = 0.3
        self.min_speed = -0.3
        
        # Variables to track current state
        self.publishing_thread = None
        self.stop_thread = False
        self.window_active = False
        
        # Command queue for non-blocking operations
        self.command_queue = queue.Queue()
        
        # Publisher timer (100Hz = 0.01s)
        self.timer_period = 0.01  # 100Hz
        self.publish_timer = self.create_timer(
            self.timer_period, 
            self.publish_callback,
            callback_group=MutuallyExclusiveCallbackGroup()
        )
        
        self.norender = norender
        if not self.norender:
            # Initialize GUI window
            self.setup_gui()
        
        # Start command processing thread
        self.command_thread = threading.Thread(target=self.process_commands)
        self.command_thread.daemon = True
        self.command_thread.start()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Robot Teleop Control")
        self.root.geometry("500x420")  # Smaller window size
        self.root.configure(bg="#f0f0f0")
        
        # Make window always on top
        self.root.attributes("-topmost", True)
        
        # Make window non-resizable
        self.root.resizable(False, False)
        
        # Set up fonts (smaller fonts)
        title_font = font.Font(family="Arial", size=14, weight="bold")
        info_font = font.Font(family="Arial", size=11)
        speed_font = font.Font(family="Courier", size=12)
        
        # Title
        tk.Label(self.root, text="Robot Teleop Control", font=title_font, bg="#f0f0f0").pack(pady=(10, 5))
        
        # Current speed display (using compact grid layout)
        self.speed_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.speed_frame.pack(pady=5)
        
        # First row: Linear X and Angular Z
        tk.Label(self.speed_frame, text="Linear X:", font=speed_font, bg="#f0f0f0").grid(row=0, column=0, sticky="e", padx=(0, 5), pady=2)
        self.linear_speed_label = tk.Label(self.speed_frame, text="0.0 m/s", font=speed_font, bg="#e6e6e6", width=10, relief="sunken", padx=3, pady=2)
        self.linear_speed_label.grid(row=0, column=1, sticky="w", pady=2, padx=2)
        
        tk.Label(self.speed_frame, text="Angular Z:", font=speed_font, bg="#f0f0f0").grid(row=0, column=2, sticky="e", padx=(10, 5), pady=2)
        self.angular_speed_label = tk.Label(self.speed_frame, text="0.0 rad/s", font=speed_font, bg="#e6e6e6", width=10, relief="sunken", padx=3, pady=2)
        self.angular_speed_label.grid(row=0, column=3, sticky="w", pady=2, padx=2)
        
        # Second row: Policy and Navigation
        tk.Label(self.speed_frame, text="Policy:", font=speed_font, bg="#f0f0f0").grid(row=1, column=0, sticky="e", padx=(0, 5), pady=2)
        self.policy_label = tk.Label(self.speed_frame, text="OFF", font=speed_font, bg="#e6e6e6", width=10, relief="sunken", padx=3, pady=2)
        self.policy_label.grid(row=1, column=1, sticky="w", pady=2, padx=2)
        
        tk.Label(self.speed_frame, text="Navigation:", font=speed_font, bg="#f0f0f0").grid(row=1, column=2, sticky="e", padx=(10, 5), pady=2)
        self.navigation_label = tk.Label(self.speed_frame, text="OFF", font=speed_font, bg="#e6e6e6", width=10, relief="sunken", padx=3, pady=2)
        self.navigation_label.grid(row=1, column=3, sticky="w", pady=2, padx=2)
        
        # Status display
        self.status_label = tk.Label(self.root, text="Status: Click window to activate", font=info_font, bg="#fff3cd", fg="#856404", padx=5, pady=4)
        self.status_label.pack(fill="x", padx=10, pady=5)
        
        # Controls instructions (more compact layout)
        controls_frame = tk.Frame(self.root, bg="#f0f0f0", relief="ridge", bd=1)
        controls_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        
        instructions = [
            ("↑/↓", "Adjust linear speed (+/- 0.1)"),
            ("←/→", "Adjust angular speed (+/- 0.1)"),
            ("Space", "Reset all speeds to zero"),
            ("q", "Quit")
        ]
        
        tk.Label(controls_frame, text="Controls:", font=info_font, bg="#f0f0f0").grid(row=0, column=0, columnspan=2, sticky="w", pady=(5, 5), padx=5)
        
        for i, (key, desc) in enumerate(instructions):
            bg_color = "#e6e6e6"
            
            # Highlight space key control
            if key == "Space":
                bg_color = "#d1ecf1"  # Light blue background for space
            
            key_label = tk.Label(controls_frame, text=key, font=speed_font, bg=bg_color, width=6, padx=3, pady=2)
            key_label.grid(row=i+1, column=0, sticky="w", padx=5, pady=3)
            
            desc_label = tk.Label(controls_frame, text=desc, font=info_font, bg="#f0f0f0", anchor="w", padx=3)
            desc_label.grid(row=i+1, column=1, sticky="w", padx=5, pady=3)
        
        # Bind events
        self.root.bind("<FocusIn>", self.on_window_focus)
        self.root.bind("<FocusOut>", self.on_window_unfocus)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Setup key bindings
        self.root.bind("<KeyPress>", self.on_key_press)
        
        # Update the GUI periodically
        self.update_gui()

    def process_commands(self):
        """Background thread to process commands from the queue"""
        while not self.stop_thread:
            try:
                # Get command from queue with a timeout to allow checking stop_thread
                command_func, args = self.command_queue.get(timeout=0.1)
                # Execute the command
                command_func(*args)
                # Mark task as done
                self.command_queue.task_done()
            except queue.Empty:
                # No commands in queue, just continue
                continue
            except Exception as e:
                self.get_logger().error(f'Error processing command: {e}')

    def enqueue_command(self, command_func, *args):
        """Add a command to the queue for processing in background thread"""
        self.command_queue.put((command_func, args))

    def update_gui(self):
        if hasattr(self, 'root') and self.root.winfo_exists():
            # Update speed labels
            self.linear_speed_label.config(text=f"{self.linear_speed_x:.1f} m/s")
            self.angular_speed_label.config(text=f"{self.angular_speed_z:.1f} rad/s")
            
            # Update policy status
            if hasattr(self, 'policy_active') and self.policy_active:
                self.policy_label.config(text="ON", bg="#d4edda", fg="#155724")
            else:
                self.policy_label.config(text="OFF", bg="#e6e6e6", fg="#000000")
            
            # Update status
            if not self.window_active:
                status_text = "Status: Click window to activate"
                bg_color = "#fff3cd"
                fg_color = "#856404"
            else:
                status_text = "Status: ACTIVE (publishing at 100Hz)"
                bg_color = "#d4edda"
                fg_color = "#155724"
            
            self.status_label.config(text=status_text, bg=bg_color, fg=fg_color)
            
            # Schedule next update (every 50ms for more responsive UI)
            self.root.after(50, self.update_gui)

    def on_window_focus(self, event):
        self.window_active = True
        self.get_logger().info('Window activated, keyboard control enabled')

    # def on_window_unfocus(self, event):
    #     self.window_active = False
    #     self.get_logger().info('Window deactivated, keyboard control disabled')
        
    #     # Safety: stop the robot when window loses focus
    #     self.linear_speed_x = 0.0
    #     self.angular_speed_z = 0.0
    
    def on_window_unfocus(self, event):
        self.window_active = False
        self.get_logger().info('Window deactivated, keyboard control disabled')
        
        # Safety: stop the robot when window loses focus
        self.linear_speed_x = 0.0
        self.angular_speed_z = 0.0
        
        # Safety: ensure navigation is deactivated
        if self.navigation_active:
            # Explicitly deactivate navigation
            self.navigation_active = False
            
            # Send navigation deactivation message
            msg = Bool()
            msg.data = False
            self.nav_pub.publish(msg)
            
            self.get_logger().info('Navigation mode forcibly disabled for safety')
            
            # Update UI if available
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.navigation_label.config(text="OFF", bg="#e6e6e6", fg="#000000")
                self.update_command_feedback("Navigation disabled (window unfocused)")
        else:
            # Even if navigation is already off, send the command again to ensure it's stopped
            msg = Bool()
            msg.data = False
            self.nav_pub.publish(msg)
        
    def on_closing(self):
        self.get_logger().info('Window closing, shutting down')
        
        # Safety: stop the robot
        stop_msg = Twist()
        self.publisher.publish(stop_msg)
        
        # Signal to exit the main loop
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.quit()
        
        self.stop_thread = True

    def on_key_press(self, event):
        if not self.window_active:
            return
            
        key = event.keysym
        
        if key == 'q':
            self.on_closing()
            return
        
        elif key == 'space':
            # Reset all speeds to zero
            self.linear_speed_x = 0.0
            self.angular_speed_z = 0.0
            self.get_logger().info('Speeds reset to zero')
            
        elif key == 'Up':
            self.linear_speed_x = min(self.linear_speed_x + self.speed_increment, self.max_speed)
            self.get_logger().info(f'Linear speed X: {self.linear_speed_x:.1f}')

        elif key == 'Down':
            self.linear_speed_x = max(self.linear_speed_x - self.speed_increment, self.min_speed)
            self.get_logger().info(f'Linear speed X: {self.linear_speed_x:.1f}')
            
        elif key == 'Left':
            self.angular_speed_z = min(self.angular_speed_z + self.speed_increment, self.max_speed)
            self.get_logger().info(f'Angular speed Z: {self.angular_speed_z:.1f}')
            
        elif key == 'Right':
            self.angular_speed_z = max(self.angular_speed_z - self.speed_increment, self.min_speed)
            self.get_logger().info(f'Angular speed Z: {self.angular_speed_z:.1f}')

    def publish_callback(self):
        if not self.window_active:
            # Don't publish when window not active
            return

        msg = Twist()

        # Initialize both linear and angular components to zero
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        # Set current speeds (allowing simultaneous linear and angular)
        msg.linear.x = self.linear_speed_x
        msg.angular.z = self.angular_speed_z
            
        self.publisher.publish(msg)

    def run(self):
        if self.norender:
            # CLI mode: listen for keypresses in the terminal
            try:
                print("\n[CLI Teleop Mode] Use your keyboard to control the robot. Press 'q' to quit.\n")
                self.print_cli_instructions()
                self.window_active = True  # Always active in CLI mode
                self.cli_keypress_loop()
            except KeyboardInterrupt:
                self.get_logger().info('Keyboard interrupt, shutting down')
            finally:
                stop_msg = Twist()
                self.publisher.publish(stop_msg)
        else:
            try:
                self.root.mainloop()
            except KeyboardInterrupt:
                self.get_logger().info('Keyboard interrupt, shutting down')
            finally:
                # Safety: stop the robot
                stop_msg = Twist()
                self.publisher.publish(stop_msg)

    def print_cli_instructions(self):
        print("================ INSTRUCTIONS ================")
        print("Arrow Up/Down : Adjust linear speed (+/- 0.1)")
        print("Arrow Left/Right : Adjust angular speed (+/- 0.1)")
        print("Space: Reset all speeds to zero")
        print("q: Quit")
        print("0/1/2: Switch predefined Z height")
        print("i/k: Increase/Decrease torso height")
        print("r: Reset robot pose and clear point cloud")
        print("c: Collect and stitch point clouds")
        print("p: Toggle policy ON/OFF")
        print("t/f: Save data (success/failure)")
        print("g: Slightly release gripper")
        print("b: Clean stitched buffer")
        print("s: Predict SIR")
        print("e: Execute SIR (torso only)")
        print("o: Generate random pose")
        print("n: Change row")
        print("m: Change column")
        print("h: Toggle navigation mode ON/OFF")  # Newly added navigation control instruction
        print("==============================================\n")

    def cli_keypress_loop(self):
        print("[INFO] Make sure your terminal window is focused. Press keys to control the robot.")
        print("[INFO] Press 'q' to quit.")
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            while True:
                ch = sys.stdin.read(1)
                if ch == '\x1b':  # Arrow keys start with ESC
                    next1 = sys.stdin.read(1)
                    if next1 == '[':
                        next2 = sys.stdin.read(1)
                        if next2 == 'A':  # Up
                            event = type('Event', (), {'keysym': 'Up'})
                            self.on_key_press(event)
                        elif next2 == 'B':  # Down
                            event = type('Event', (), {'keysym': 'Down'})
                            self.on_key_press(event)
                        elif next2 == 'C':  # Right
                            event = type('Event', (), {'keysym': 'Right'})
                            self.on_key_press(event)
                        elif next2 == 'D':  # Left
                            event = type('Event', (), {'keysym': 'Left'})
                            self.on_key_press(event)
                elif ch == ' ':  # Space
                    event = type('Event', (), {'keysym': 'space'})
                    self.on_key_press(event)
                elif ch.lower() == 'q':
                    event = type('Event', (), {'keysym': 'q'})
                    self.on_key_press(event)
                    print("[INFO] Quit command received. Exiting...")
                    break
                elif ch in ['0', '1', '2', 'i', 'k', 'r', 'c', 'p', 't', 'f', 'g', 'b', 's', 'e', 'o', 'n', 'm', 'h',
                            'I', 'K', 'R', 'C', 'P', 'T', 'F', 'G', 'B', 'S', 'E', 'O', 'N', 'M', 'H']:
                    event = type('Event', (), {'keysym': ch})
                    self.on_key_press(event)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# Client Rollout Manager class extending ArrowKeyTeleop
class ClientRolloutManager(ArrowKeyTeleop):
    def __init__(self, norender=False):
        super().__init__(norender=norender)
        self.get_logger().info('Starting Client Rollout Manager')

        # Override the node name
        self.get_logger().info('Node name: %s' % self.get_name())

        self.torso_idx = None
        self.policy_active = False
        self.torso_qpos = [0.00000000e+00,  7.85394366e-01, -1.57079253e+00,  7.85398163e-01,  0.00000000e+00,  0.00000000e+00]
        self.torso_min = 0.17
        self.torso_max = 1.22
        self.torso1 = None
        
        # Add navigation enable flag
        self.navigation_active = False
        
        # Create publishers for different functions
        self.reset_pub = self.create_publisher(Empty, '/manager/reset_robot', 1) 
        self.collect_pc_pub = self.create_publisher(Empty, '/manager/collect_pointcloud', 1)  
        self.policy_pub = self.create_publisher(Bool, '/manager/call_policy', 1)  
        self.save_pub = self.create_publisher(Bool, '/manager/save_data', 1)
        self.control_pub = self.create_publisher(RobotControl, '/robot/control', 1) 
        self.clean_buffer_pub = self.create_publisher(Empty, '/manager/clean_stitched_buffer', 1)
        self.sir_pub = self.create_publisher(Empty, '/manager/call_SIR', 1)
        self.sir_execute_pub = self.create_publisher(Empty, '/manager/SIR_execute', 1)
        self.sample_pose_pub = self.create_publisher(Empty, '/manager/sample_pose', 1)
        self.row_pub = self.create_publisher(Empty, '/manager/row', 1)
        self.col_pub = self.create_publisher(Empty, '/manager/col', 1)
        
        # Add navigation control publisher
        self.nav_pub = self.create_publisher(Bool, '/manager/is_nav', 1)

        self.robot_state_sub = self.create_subscription(RobotState, '/robot/state', self.robot_state_callback, 1)
        # Update controls instructions
        if not self.norender:
            self.update_controls_instructions()
        
        # Non-blocking feedback indicators
        self.command_feedback = {}

    def robot_state_callback(self, RobotState):
        self.torso1 = RobotState.torso_qpos[1]

    def update_controls_instructions(self):
        # Update the instruction frame with new controls
        if hasattr(self, 'root') and self.root.winfo_exists():
            # Remove old controls frame
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Frame) and widget != self.speed_frame:
                    widget.destroy()

            # Create new controls frame
            controls_frame = tk.Frame(self.root, bg="#f0f0f0", relief="ridge", bd=1)
            controls_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))
            
            info_font = font.Font(family="Arial", size=11)
            speed_font = font.Font(family="Courier", size=12)

            # Use compact two-column layout to display all control instructions
            instructions = [
                ("↑/↓", "Linear Speed (+/-)"),
                ("←/→", "Angular Speed (+/-)"),
                ("0/1/2", "Switch torso height"),
                ("i/k", "Inc/Dec torso height"),
                ("r/rr", "Reset arm/controller"),
                ("c", "Collect Scan"),
                ("p", "On/OFF Policy"),
                ("h", "ON/OFF Navigation"),
                ("t/f", "Save Data"),
                ("g", "Release Gripper"),
                ("b", "Clean Buffer"),
                ("s", "Predict SIR"),
                ("e", "Execute Torso"),
                ("o", "Random Pose"),
                ("n", "Change Row"),
                ("m", "Change Column"),
                ("Space", "Reset base speed"),
                ("q", "Quit")
            ]

            tk.Label(controls_frame, text="Controls:", font=info_font, bg="#f0f0f0").grid(row=0, column=0, columnspan=4, sticky="w", pady=(5, 5), padx=5)
            
            # Split instructions into two columns to make UI more compact
            half = len(instructions) // 2 + len(instructions) % 2
            for i, (key, desc) in enumerate(instructions[:half]):
                bg_color = "#e6e6e6"
                
                # Highlight important keys
                if key in ["r", "c", "p", "h", "t", "t/f", "g", "b", "s", "e", "o", "n", "m"]:
                    bg_color = "#d1ecf1"  # Light blue background
                
                key_label = tk.Label(controls_frame, text=key, font=speed_font, bg=bg_color, width=6, padx=3, pady=2)
                key_label.grid(row=i+1, column=0, sticky="w", padx=3, pady=2)
                
                desc_label = tk.Label(controls_frame, text=desc, font=info_font, bg="#f0f0f0", anchor="w", width=22)
                desc_label.grid(row=i+1, column=1, sticky="w", padx=0, pady=2)
            
            # Right column
            for i, (key, desc) in enumerate(instructions[half:]):
                bg_color = "#e6e6e6"
                
                # Highlight important keys
                if key in ["r", "c", "p", "h", "t", "t/f", "g", "b", "s", "e", "o", "n", "m"]:
                    bg_color = "#d1ecf1"  # Light blue background
                
                key_label = tk.Label(controls_frame, text=key, font=speed_font, bg=bg_color, width=6, padx=3, pady=2)
                key_label.grid(row=i+1, column=2, sticky="w", padx=3, pady=2)
                
                desc_label = tk.Label(controls_frame, text=desc, font=info_font, bg="#f0f0f0", anchor="w", width=20)
                desc_label.grid(row=i+1, column=3, sticky="w", padx=0, pady=2)
                
            # Add feedback area (reduced padding)
            feedback_frame = tk.Frame(controls_frame, bg="#f0f0f0")
            feedback_frame.grid(row=len(instructions)//2+2, column=0, columnspan=4, sticky="ew", padx=5, pady=(10, 3))
            
            tk.Label(feedback_frame, text="Command Status:", font=info_font, bg="#f0f0f0").pack(anchor="w")
            self.feedback_label = tk.Label(feedback_frame, text="Ready", font=speed_font, bg="#e6e6e6", padx=3, pady=2, anchor="w")
            self.feedback_label.pack(fill="x", anchor="w", pady=(3, 0))

    def update_command_feedback(self, message):
        """Update feedback message in UI thread-safely"""
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, lambda: self.feedback_label.config(text=message))
        elif self.norender:
            print(f"[FEEDBACK] {message}")
    
    # Command execution methods that will run in the background thread
    def execute_switch_torso(self, key):
        self.update_command_feedback(f"Switching to Z-index: {key}...")
        msg = RobotControl()
        msg.command_type = "torso"
        if key == '0':  
            msg.torso_qpos = INIT_POSE["torso_top"].tolist()
        elif key == '1':
            msg.torso_qpos = INIT_POSE["torso_mid"].tolist()
        elif key == '2':
            msg.torso_qpos = INIT_POSE["torso_bottom"].tolist()
        self.control_pub.publish(msg)
        self.get_logger().info(f'Switched to Z-index: {key}')
        self.update_command_feedback(f"Torso position set to {key}")
        
    def execute_release_gripper(self):
        self.update_command_feedback("Releasing gripper...")
        msg = RobotControl()
        msg.command_type = "gripper"
        msg.right_gripper_qpos_normalized = 0.13 # release gripper
        self.control_pub.publish(msg)
        self.get_logger().info('Released gripper')
        self.update_command_feedback("Gripper released")
        
    def execute_reset_robot(self):
        self.update_command_feedback("Resetting robot, please wait...")
        msg = RobotControl()
        msg.command_type = "right_arm"
        msg.right_arm_qpos = INIT_POSE["right_arm"].tolist()
        msg.right_gripper_qpos_normalized = 0.0 # close gripper
        self.control_pub.publish(msg)
        self.reset_pub.publish(Empty())
        self.get_logger().info('Robot reset completed')
        self.update_command_feedback("Robot reset completed")
        
    def execute_collect_pointcloud(self):
        self.update_command_feedback("Collecting point cloud...")
        self.collect_pc_pub.publish(Empty())
        self.get_logger().info('Collecting and stitching point clouds')
        self.update_command_feedback("Point cloud collection initiated")
        
    def execute_toggle_policy(self):
        msg = Bool()
        if not self.policy_active:
            msg.data = True
            self.update_command_feedback("Starting policy...")
            self.get_logger().info('Called the policy')
        else:
            msg.data = False
            self.update_command_feedback("Stopping policy...")
            self.get_logger().info('Stopped the policy')
        self.policy_pub.publish(msg)
        self.policy_active = not self.policy_active
        self.update_command_feedback(f"Policy {'activated' if self.policy_active else 'deactivated'}")
    
    def execute_toggle_navigation(self):
        """Toggle navigation function status"""
        self.navigation_active = not self.navigation_active
        
        # Publish navigation status message
        msg = Bool()
        msg.data = self.navigation_active
        self.nav_pub.publish(msg)
        
        status = "ON" if self.navigation_active else "OFF"
        self.get_logger().info(f'Navigation mode: {status}')
        self.update_command_feedback(f"Navigation {status}")
        
        # Update UI display
        if hasattr(self, 'root') and self.root.winfo_exists():
            if self.navigation_active:
                self.navigation_label.config(text="ON", bg="#d4edda", fg="#155724")
            else:
                self.navigation_label.config(text="OFF", bg="#e6e6e6", fg="#000000")
        
    def execute_save_data(self, success):
        status = "Success" if success else "Failed"
        self.update_command_feedback(f"Saving data ({status})...")
        msg = Bool()
        msg.data = success
        self.save_pub.publish(msg)
        self.get_logger().info(f'Rollout result: {status}, Saving PCL and robot pose')
        self.update_command_feedback(f"Data saved ({status})")

    def execute_clean_buffer(self):
        self.update_command_feedback("Cleaning stitched buffer...")
        self.clean_buffer_pub.publish(Empty())
        self.get_logger().info('Cleaned stitched buffer')
        self.update_command_feedback("Stitched buffer cleaned")

    def execute_predict_sir(self):
        self.update_command_feedback("Predicting SIR...")
        self.sir_pub.publish(Empty())
        self.get_logger().info('Predicting SIR')
        self.update_command_feedback("SIR prediction initiated")
        
    def execute_sir_execute(self):
        self.update_command_feedback("Executing (Torso Only)...")
        self.sir_execute_pub.publish(Empty())
        self.get_logger().info('Executing SIR (Torso Only)')
        self.update_command_feedback("Torso execution initiated")
        
    def execute_sample_pose(self):
        self.update_command_feedback("Generating random pose...")
        self.sample_pose_pub.publish(Empty())
        self.get_logger().info('Generating random pose')
        self.update_command_feedback("Random pose generation initiated")
        
    def execute_change_row(self):
        self.update_command_feedback("Changing row...")
        self.row_pub.publish(Empty())
        self.get_logger().info('Changing row')
        self.update_command_feedback("Row changed")
        
    def execute_change_col(self):
        self.update_command_feedback("Changing column...")
        self.col_pub.publish(Empty())
        self.get_logger().info('Changing column')
        self.update_command_feedback("Column changed")

    def execute_adjust_torso1(self, direction):
        if self.torso1 is None:
            self.update_command_feedback("Torso1 value not available yet")
            return
            
        adjustment = 0.02 * direction
        new_value = self.torso1 + adjustment
        
        # Clamp to min/max values
        new_value = max(min(new_value, self.torso_max), self.torso_min)
        self.update_command_feedback(f"Adjusting torso1 to {new_value:.2f}...")
        
        msg = RobotControl()
        msg.command_type = "torso"
        # Create a copy of the current torso position and modify the torso1 value
        torso_qpos = self.torso_qpos.copy()
        torso_qpos[1] = new_value
        torso_qpos[2] = -2*new_value
        torso_qpos[3] = new_value
        msg.torso_qpos = torso_qpos
        
        self.control_pub.publish(msg)
        # self.get_logger().info(f'Adjusted torso1 to {new_value:.2f}')
        # self.update_command_feedback(f"Torso1 adjusted to {new_value:.2f}")

    def on_key_press(self, event):
        if not self.window_active:
            return
            
        key = event.keysym
        
        # If navigation mode is active, don't handle arrow keys (let navigation node handle them)
        if self.navigation_active and key in ['Up', 'Down', 'Left', 'Right']:
            return
        
        # Call the parent class's key press handler for existing functionality
        if key in ['q', 'space', 'Up', 'Down', 'Left', 'Right']:
            super().on_key_press(event)
            return
        
        # Handle new keys with non-blocking operations by using command queue
        elif key in ['0', '1', '2']:
            # Switch predefined Z height (enqueue command)
            self.linear_speed_x = 0.0
            self.angular_speed_z = 0.0
            self.enqueue_command(self.execute_switch_torso, key)
            self.update_command_feedback(f"Processing: Switch torso to {key}...")

        elif key.lower() == 'i':
            # Decrease torso1 value by 0.1
            self.enqueue_command(self.execute_adjust_torso1, -1)
            self.update_command_feedback("Processing: Decrease torso height...")
            
        elif key.lower() == 'k':
            # Increase torso1 value by 0.1
            self.enqueue_command(self.execute_adjust_torso1, 1)
            self.update_command_feedback("Processing: Increase torso height...")

        elif key.lower() == 'g':
            # Release gripper (enqueue command)
            self.linear_speed_x = 0.0
            self.angular_speed_z = 0.0
            self.enqueue_command(self.execute_release_gripper)
            self.update_command_feedback("Processing: Release gripper...")
            
        elif key.lower() == 'r':
            # Reset robot pose and clear point cloud (enqueue command)
            self.linear_speed_x = 0.0
            self.angular_speed_z = 0.0
            self.enqueue_command(self.execute_reset_robot)
            self.update_command_feedback("Processing: Reset robot...")

        elif key.lower() == 'c':
            # Collect and stitch point clouds (enqueue command)
            self.enqueue_command(self.execute_collect_pointcloud)
            self.update_command_feedback("Processing: Collect point cloud...")

        elif key.lower() == 'p':
            # Call the policy (enqueue command)
            self.enqueue_command(self.execute_toggle_policy)
            status = "OFF" if self.policy_active else "ON"
            self.update_command_feedback(f"Processing: Set policy to {status}...")
            
        elif key.lower() == 'h':
            # set to zero for safty
            self.linear_speed_x = 0.0
            self.angular_speed_z = 0.0
            # Toggle navigation mode
            self.enqueue_command(self.execute_toggle_navigation)
            status = "ON" if not self.navigation_active else "OFF"
            self.update_command_feedback(f"Processing: Set navigation to {status}...")

        elif key.lower() == 's':
            # Predict SIR (enqueue command)
            self.enqueue_command(self.execute_predict_sir)
            self.update_command_feedback("Processing: Predict SIR...")
            
        elif key.lower() == 'e':
            # Execute SIR (torso only) (enqueue command)
            self.enqueue_command(self.execute_sir_execute)
            self.update_command_feedback("Processing: Execute (Torso Only)...")
            
        elif key.lower() == 'o':
            # Generate random pose (enqueue command)
            self.enqueue_command(self.execute_sample_pose)
            self.update_command_feedback("Processing: Generate Random Pose...")

        elif key.lower() == 't' or key.lower() == 'f':
            # Save point cloud and other data (enqueue command)
            success = (key.lower() == 't')
            self.enqueue_command(self.execute_save_data, success)
            result = "success" if success else "failure"
            self.update_command_feedback(f"Processing: Save data ({result})...")

        elif key.lower() == 'b':
            # Clean stitched buffer (enqueue command)
            self.enqueue_command(self.execute_clean_buffer)
            self.update_command_feedback("Processing: Clean stitched buffer...")
            
        elif key.lower() == 'n':
            # Change row (enqueue command)
            self.enqueue_command(self.execute_change_row)
            self.update_command_feedback("Processing: Change row...")
            
        elif key.lower() == 'm':
            # Change column (enqueue command)
            self.enqueue_command(self.execute_change_col)
            self.update_command_feedback("Processing: Change column...")

    def publish_callback(self):
        # If navigation mode is enabled, don't publish control commands
        if self.navigation_active:
            return
            
        # Original publishing logic
        if not self.window_active:
            # Don't publish when window not active
            return

        msg = Twist()

        # Initialize both linear and angular components to zero
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        # Set current speeds (allowing simultaneous linear and angular)
        msg.linear.x = self.linear_speed_x
        msg.angular.z = self.angular_speed_z
            
        self.publisher.publish(msg)

    def run(self):
        if self.norender:
            # CLI mode: listen for keypresses in the terminal
            try:
                print("\n[CLI Teleop Mode] Use your keyboard to control the robot. Press 'q' to quit.\n")
                self.print_cli_instructions()
                self.window_active = True  # Always active in CLI mode
                self.cli_keypress_loop()
            except KeyboardInterrupt:
                self.get_logger().info('Keyboard interrupt, shutting down')
            finally:
                stop_msg = Twist()
                self.publisher.publish(stop_msg)
        else:
            try:
                self.root.mainloop()
            except KeyboardInterrupt:
                self.get_logger().info('Keyboard interrupt, shutting down')
            finally:
                # Safety: stop the robot
                stop_msg = Twist()
                self.publisher.publish(stop_msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--norender', action='store_true', default=False, help='No GUI display, command line mode')
    args = parser.parse_args()

    rclpy.init()
    node = ClientRolloutManager(norender=args.norender)
    
    # Use a separate thread for spinning the node
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    
    spin_thread = threading.Thread(target=executor.spin)
    spin_thread.daemon = True
    spin_thread.start()
    
    def signal_handler(sig, frame):
        print("\nReceived Ctrl+C, shutting down...")
        if hasattr(node, 'root') and not node.norender and node.root.winfo_exists():
            node.root.quit()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        node.run()
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()