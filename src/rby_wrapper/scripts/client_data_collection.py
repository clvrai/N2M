#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, String
from rby_wrapper.msg import RobotState as RobotStateMsg
import cv2
import numpy as np
import h5py
import os
from datetime import datetime
import time
from typing import Dict, List, Optional
import threading
from queue import Queue
from pynput import keyboard
import sys
import argparse
import shutil
from termcolor import colored

class TerminalGUI:
    def __init__(self):
        self.width = shutil.get_terminal_size().columns
        self.height = 12
        # Calculate box width based on progress bar length
        self.box_width = min(70, self.width - 4)  # 50 (progress bar) + 10 (padding) + 10 (percentage)
        self.clear_screen()
        
    def clear_screen(self):
        """Clear the terminal screen"""
        print('\033[2J\033[H', end='')
        
    def move_cursor(self, x: int, y: int):
        """Move cursor to position (x, y)"""
        print(f'\033[{y};{x}H', end='')
        
    def print_at(self, x: int, y: int, text: str, color: str = None):
        """Print text at position (x, y) with optional color"""
        self.move_cursor(x, y)
        if color:
            print(f'\033[{color}m{text}\033[0m', end='')
        else:
            print(text, end='')
        
    def draw_box(self, x1: int, y1: int, x2: int, y2: int):
        """Draw a box using ASCII characters"""
        # Draw top border
        self.print_at(x1, y1, '┌' + '─' * (x2 - x1 - 1) + '┐')
        # Draw side borders
        for y in range(y1 + 1, y2):
            self.print_at(x1, y, '│' + ' ' * (x2 - x1 - 1) + '│')
        # Draw bottom border
        self.print_at(x1, y2, '└' + '─' * (x2 - x1 - 1) + '┘')
        
    def draw_progress_bar(self, x: int, y: int, width: int, progress: float):
        """Draw a progress bar"""
        # Make progress bar shorter to fit within the box
        bar_width = min(width - 10, 50)  # Limit maximum width to 50 characters
        filled = int(bar_width * progress)
        bar = '█' * filled + '░' * (bar_width - filled)
        self.print_at(x, y, f'[{bar}] {progress*100:.1f}%')
        
    def update_display(self, elapsed_time: float, frames: int, topic_rates: Dict[str, float], data_collection_rate: float, max_horizon: int, waiting_for_label: bool = False):
        """Update the entire display"""
        self.clear_screen()
        
        # Set box position to start from left with small margin
        x1 = 2
        x2 = x1 + self.box_width
        
        # Draw main box
        self.draw_box(x1, 1, x2, self.height)
        
        # Draw title
        self.print_at(x1 + 2, 2, "Data Collection Status")
        
        # Draw progress bar
        progress = min(1.0, frames / max_horizon)  # Use frames/max_horizon for progress
        self.draw_progress_bar(x1 + 2, 3, self.box_width - 4, progress)
        
        # Draw time and frames
        self.print_at(x1 + 2, 4, f"Time: {elapsed_time:.1f}s | Frames: {frames}/{max_horizon}")
        
        # Draw data collection rate
        self.print_at(x1 + 2, 5, f"Data Collection Rate: {data_collection_rate:.1f} Hz")
        
        # Draw topic rates
        y = 6
        for topic, rate in topic_rates.items():
            self.print_at(x1 + 2, y, f"{topic}: {rate:.1f} Hz")
            y += 1
            
        # Draw instructions with color if waiting for label
        if waiting_for_label:
            self.print_at(x1 + 2, self.height - 2, "Press T/t for success, F/f for failure", "33")  # Yellow color
        else:
            self.print_at(x1 + 2, self.height - 2, "Press T/t for success, F/f for failure")
        
        # Flush output
        sys.stdout.flush()

    def clear_and_reset(self):
        """Clear the screen and reset cursor position"""
        self.clear_screen()
        self.move_cursor(1, 1)

class DataCollectionNode(Node):
    def __init__(self, task_name: str, max_horizon: int):
        super().__init__('data_collection_node')
        
        # Store task name and max horizon
        self.task_name = task_name
        self.max_horizon = max_horizon
        
        # Data collection parameters
        self.data_collection_rate = 30.0  # Hz
        self.left_cam_rate = 30.0        # Hz
        self.right_cam_rate = 30.0       # Hz
        self.head_cam_rate = 30.0        # Hz
        self.robot_state_rate = 100.0    # Hz
        
        self.callback_counters = {
            'left_cam': 0,
            'right_cam': 0,
            'head_cam': 0,
            'robot_state': 0,
            'data_collection': 0
        }
        
        # Create data directory if it doesn't exist
        self.base_dir = os.path.expanduser('/home/mm/workbench/real/dataset')
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize episode counter based on existing data
        self.episode_counter = self.get_next_episode_number()
        
        # Collection progress
        self.collection_start_time = None
        self.collection_end_time = None
        self.progress_timer = None
        
        # Topic rate monitoring
        self.topic_rates = {
            'Left Camera': 0.0,
            'Right Camera': 0.0,
            'Head Camera': 0.0,
            'Robot State': 0.0
        }
        
        # Topic rate calculation
        self.topic_timestamps = {
            'Left Camera': [],
            'Right Camera': [],
            'Head Camera': [],
            'Robot State': []
        }
        self.rate_window = 2.0  # Calculate rate over 2 seconds window

        # Topic rate calculation lock
        self.rate_lock = threading.Lock()
        
        # Start topic rate calculation thread
        self.rate_calculation_running = True
        self.rate_calculation_thread = threading.Thread(target=self.calculate_topic_rates)
        self.rate_calculation_thread.daemon = True
        self.rate_calculation_thread.start()
        
        # Terminal GUI
        self.gui = TerminalGUI()
        
        # Collection state
        self.is_collecting = False
        self.waiting_for_label = False
        
        # Initialize data storage
        self.current_data = {
            'left_cam': None,
            'right_cam': None,
            'head_cam': None,
            'robot_state': None
        }
        
        # Initialize data collection state
        self.collected_data = {
            'left_cam': [],
            'right_cam': [],
            'head_cam': [],
            'robot_state': []
        }
        
        # Setup subscribers
        self.setup_subscribers()
        
        # Setup publishers for start and done signals
        self.start_pub = self.create_publisher(Bool, '/collection/start', 10)
        self.done_pub = self.create_publisher(Bool, '/collection/done', 10)
        
        # Setup timer for data collection
        self.timer = self.create_timer(1.0/self.data_collection_rate, self.data_collection_callback)
        
        # verbose for debug
        self.verbose_debug = False 
        
        # Initialize keyboard monitoring
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()
        
        self.get_logger().info('Data collection node initialized')
        self.get_logger().info(f'Starting from episode {self.episode_counter}')
        self.get_logger().info(f'Maximum horizon: {self.max_horizon} steps')
        self.get_logger().info('Key bindings:')
        self.get_logger().info('  - Press S/s to start collecting data')
        self.get_logger().info('  - Press T/t to stop and mark as success')
        self.get_logger().info('  - Press F/f to stop and mark as failure')

    def get_next_episode_number(self) -> int:
        """Get the next episode number based on existing data"""
        task_dir = os.path.join(self.base_dir, self.task_name)
        if not os.path.exists(task_dir):
            return 0
            
        # Get all episode directories
        episode_dirs = [d for d in os.listdir(task_dir) 
                       if os.path.isdir(os.path.join(task_dir, d)) and d.isdigit()]
        
        if not episode_dirs:
            return 0
            
        # Get the maximum episode number
        max_episode = max(int(d) for d in episode_dirs)
        return max_episode + 1

    def setup_subscribers(self):
        """Setup all subscribers with appropriate QoS profiles"""
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Camera subscribers
        self.left_cam_sub = self.create_subscription(
            CompressedImage,
            '/left_camera/left_camera/color/image_rect_raw/compressed',
            self.left_cam_callback,
            qos_profile
        )
        
        self.right_cam_sub = self.create_subscription(
            CompressedImage,
            '/right_camera/right_camera/color/image_rect_raw/compressed',
            self.right_cam_callback,
            qos_profile
        )
        
        self.head_cam_sub = self.create_subscription(
            CompressedImage,
            '/head_camera/head_camera/color/image_raw/compressed',
            self.head_cam_callback,
            qos_profile
        )
        
        # Robot state subscriber
        self.robot_state_sub = self.create_subscription(
            RobotStateMsg,
            '/robot/state',
            self.robot_state_callback,
            qos_profile
        )

    def calculate_topic_rates(self):
        """Thread function to calculate topic rates periodically"""
        while self.rate_calculation_running:
            current_time = time.time()
            
            with self.rate_lock:
                for topic_name in self.topic_timestamps:
                    timestamps = self.topic_timestamps[topic_name]
                    
                    # Remove timestamps older than the window
                    while timestamps and current_time - timestamps[0] > self.rate_window:
                        timestamps.pop(0)
                    
                    # Calculate rate
                    if len(timestamps) > 1:
                        rate = (len(timestamps) - 1) / (current_time - timestamps[0]) if current_time > timestamps[0] else 0.0
                        self.topic_rates[topic_name] = rate
                    elif len(timestamps) == 0:
                        # No messages received in the window
                        self.topic_rates[topic_name] = 0.0
            
            # Sleep for a short time before recalculating
            time.sleep(0.1)

    def left_cam_callback(self, msg: CompressedImage):
        """Callback for left camera images"""
        self.current_data['left_cam'] = self.convert_image(msg)
        with self.rate_lock:
            self.topic_timestamps['Left Camera'].append(time.time())
        self.callback_counters['left_cam'] += 1

    def right_cam_callback(self, msg: CompressedImage):
        """Callback for right camera images"""
        self.current_data['right_cam'] = self.convert_image(msg)
        with self.rate_lock:
            self.topic_timestamps['Right Camera'].append(time.time())
        self.callback_counters['right_cam'] += 1

    def head_cam_callback(self, msg: CompressedImage):
        """Callback for head camera images"""
        self.current_data['head_cam'] = self.convert_image(msg)
        with self.rate_lock:
            self.topic_timestamps['Head Camera'].append(time.time())
        self.callback_counters['head_cam'] += 1

    def robot_state_callback(self, msg: RobotStateMsg):
        """Callback for robot state data"""
        self.current_data['robot_state'] = msg
        with self.rate_lock:
            self.topic_timestamps['Robot State'].append(time.time())
        self.callback_counters['robot_state'] += 1

    def convert_image(self, msg: CompressedImage) -> np.ndarray:
        """Convert ROS CompressedImage message to numpy array"""
        try:
            # Convert CompressedImage message to numpy array
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            self.get_logger().error(f'Error converting compressed image: {str(e)}')
            return None

    def data_collection_callback(self):
        """Main data collection callback"""
        if self.is_collecting:  # Only check is_collecting flag
            self.callback_counters['data_collection'] += 1
            # Store current data
            all_key_not_none = True
            for key in ['right_cam', 'head_cam', 'robot_state']:    # do not include left_cam
                if self.current_data[key] is None:
                    all_key_not_none = False
                    break
            if all_key_not_none:
                for key in self.current_data:
                    self.collected_data[key].append(self.current_data[key])
                
            if self.verbose_debug:
                # Print callback counters
                print(f"\nCallback counts - Collection: {self.callback_counters['data_collection']}, "
                    f"Left cam: {self.callback_counters['left_cam']}, "
                    f"Right cam: {self.callback_counters['right_cam']}, "
                    f"Head cam: {self.callback_counters['head_cam']}, "
                    f"Robot state: {self.callback_counters['robot_state']}")
            
                # Print current data status
                print(f"Current data status - Left cam: {'✓' if self.current_data['left_cam'] is not None else '✗'}, "
                    f"Right cam: {'✓' if self.current_data['right_cam'] is not None else '✗'}, "
                    f"Head cam: {'✓' if self.current_data['head_cam'] is not None else '✗'}, "
                    f"Robot state: {'✓' if self.current_data['robot_state'] is not None else '✗'}")
            
                # Print collected data lengths
                print(f"Collected data lengths - Left cam: {len(self.collected_data['left_cam'])}, "
                    f"Right cam: {len(self.collected_data['right_cam'])}, "
                    f"Head cam: {len(self.collected_data['head_cam'])}, "
                    f"Robot state: {len(self.collected_data['robot_state'])}")
            
            # Check if we've reached max_horizon
            if len(self.collected_data['left_cam']) >= self.max_horizon:
                self.get_logger().info(f'Reached maximum horizon ({self.max_horizon} steps)')
                self.is_collecting = False
                self.waiting_for_label = True
                self.collection_end_time = time.time()
                print("\nStop collection, press T/t for success or F/f for failure to give the label")

    def on_key_press(self, key):
        try:
            if hasattr(key, 'char') and key.char.lower() == 's' and not self.is_collecting and not self.waiting_for_label:
                self.start_collection()
            elif (self.is_collecting or self.waiting_for_label) and hasattr(key, 'char') and key.char.lower() in ['t', 'f']:
                success = key.char.lower() == 't'
                self.waiting_for_label = False
                self.stop_collection(success)
        except AttributeError:
            pass  # Ignore special keys

    def update_progress(self):
        """Update and display collection progress"""
        if not self.is_collecting and not self.waiting_for_label:
            return
            
        # Calculate elapsed time
        if self.collection_end_time is not None:
            elapsed_time = self.collection_end_time - self.collection_start_time
        else:
            elapsed_time = time.time() - self.collection_start_time
            
        frames = len(self.collected_data['left_cam'])
        
        # Update terminal GUI
        self.gui.update_display(elapsed_time, frames, self.topic_rates, self.data_collection_rate, self.max_horizon, self.waiting_for_label)

    def start_collection(self):
        """Start data collection for a new episode"""
        
        # Publish collection start signal
        self.start_pub.publish(Bool(data=True))
        self.get_logger().info('Published start signal to /collection/start')
        time.sleep(1.0) # wait for teleoperation.py to start
        
        # reset counters
        for key in self.callback_counters:
            self.callback_counters[key] = 0
            
        self.is_collecting = True
        self.waiting_for_label = False
        self.collection_end_time = None
        self.collected_data = {
            'left_cam': [],
            'right_cam': [],
            'head_cam': [],
            'robot_state': []
        }
        self.current_data = {
            'left_cam': None,
            'right_cam': None,
            'head_cam': None,
            'robot_state': None
        }
        # Reset topic timestamps
        with self.rate_lock:
            for topic in self.topic_timestamps:
                self.topic_timestamps[topic] = []
                self.topic_rates[topic] = 0.0
            
        self.episode_id = str(self.episode_counter)
        self.collection_start_time = time.time()
        
        # Start progress timer
        self.progress_timer = self.create_timer(0.1, self.update_progress)  # Update more frequently
        
        self.get_logger().info(f'Started collecting data for task: {self.task_name}, episode: {self.episode_id}')
        self.get_logger().info(f'Will automatically stop after {self.max_horizon} steps')

    def stop_collection(self, success: bool):
        """Stop data collection and save the data"""
        self.is_collecting = False
        
        if self.progress_timer:
            self.destroy_timer(self.progress_timer)
            self.progress_timer = None
            
        # Publish collection done signal
        self.done_pub.publish(Bool(data=True))
        self.get_logger().info('Published done signal to /collection/done')
            
        # Clear terminal GUI and reset cursor
        self.gui.clear_and_reset()
        if success:
            # Print saving information
            print("\n Success! saving data...")
            time.sleep(1.0)
            
            # Create episode directory
            episode_dir = os.path.join(self.base_dir, self.task_name, self.episode_id)
            os.makedirs(episode_dir, exist_ok=True)
            
            # Save videos
            self.save_videos(episode_dir)
            
            # Save HDF5 file
            self.save_hdf5(episode_dir, success)
            
            print(f'Data saved to {episode_dir}')
            
            # Increment episode counter
            self.episode_counter += 1
            
        else:
            print("\n Failed. discard data...")
            time.sleep(1.0)
            
        try:
            realname, row, col = self.task_name.split('_')
            print(f"\033[31m realname: {realname},row: {row}, col: {col}, Collected Num: {self.episode_counter} , \033[0m")
        except ValueError:
            print("wrong format")
            return
        print(colored('Press S/s to start next episode', 'green'))
        # Reset all collection states
        self.is_collecting = False
        self.waiting_for_label = False
        self.collection_start_time = None
        self.collection_end_time = None
        self.collected_data = {
            'left_cam': [],
            'right_cam': [],
            'head_cam': [],
            'robot_state': []
        }
        # Reset topic timestamps
        with self.rate_lock:
            for topic in self.topic_timestamps:
                self.topic_timestamps[topic] = []
                self.topic_rates[topic] = 0.0

    def save_videos(self, episode_dir: str):
        """Save collected camera data as videos"""
        try:
            # Create video writers for each camera that has data
            writers = {}
            
            # Check each camera （do not include left_cam. Maybe use a variable to control this will be better）
            for cam_name in ['right_cam', 'head_cam']:
                if len(self.collected_data[cam_name]) > 0:
                    height, width = self.collected_data[cam_name][0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writers[cam_name] = cv2.VideoWriter(
                        os.path.join(episode_dir, f'video_{cam_name.split("_")[0]}.mp4'),
                        fourcc,
                        self.data_collection_rate,
                        (width, height)
                    )
                    self.get_logger().info(
                        f'Created video writer for {cam_name} with {len(self.collected_data[cam_name])} frames '
                        f'(resolution: {width}x{height})'
                    )
                else:
                    self.get_logger().warn(f'No data collected for {cam_name}')
            
            if not writers:
                self.get_logger().warn('No camera data to save')
                return
                
            # Write frames
            max_frames = max(len(self.collected_data[cam]) for cam in writers.keys())
            for i in range(max_frames):
                for cam_name, writer in writers.items():
                    if i < len(self.collected_data[cam_name]) and self.collected_data[cam_name][i] is not None:
                        writer.write(cv2.cvtColor(self.collected_data[cam_name][i], cv2.COLOR_RGB2BGR))
            
            # Release video writers
            for writer in writers.values():
                writer.release()
            
            self.get_logger().info('Videos saved successfully')
        except Exception as e:
            self.get_logger().error(f'Error saving videos: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def save_hdf5(self, episode_dir: str, success: bool):
        """Save collected data as HDF5 file"""
        try:
            h5_path = os.path.join(episode_dir, 'data.h5')
            with h5py.File(h5_path, 'w') as f:
                # Create metadata group
                metadata = f.create_group('metadata')
                metadata.attrs['taskName'] = self.task_name
                metadata.attrs['timestamp'] = datetime.now().isoformat()
                metadata.attrs['success'] = 1 if success else 0
                metadata.attrs['trajectoryNum'] = self.episode_counter+1    # because episode_counter starts from 0
                
                if len(self.collected_data['robot_state']) > 0:
                    # Create data for each robot part
                    robot_parts = ['left_arm', 'right_arm', 'torso', 'mobility', 'left_gripper', 'right_gripper']
                    for part in robot_parts:
                        # Create group for each part
                        part_group = f.create_group(part)
                        # Create datasets for each part
                        part_group.create_dataset('qpos', data=np.array([getattr(state, f'{part}_qpos') for state in self.collected_data['robot_state']]))
                        part_group.create_dataset('target_qpos', data=np.array([getattr(state, f'{part}_target_qpos') for state in self.collected_data['robot_state']]))
                        
                        if part in ['left_gripper', 'right_gripper']:
                            part_group.create_dataset('min_qpos', data=np.array([getattr(state, f'{part}_min_qpos') for state in self.collected_data['robot_state']]))
                            part_group.create_dataset('max_qpos', data=np.array([getattr(state, f'{part}_max_qpos') for state in self.collected_data['robot_state']]))
                            part_group.create_dataset('qpos_normalized', data=np.array([getattr(state, f'{part}_qpos_normalized') for state in self.collected_data['robot_state']]))
                            part_group.create_dataset('target_qpos_normalized', data=np.array([getattr(state, f'{part}_target_qpos_normalized') for state in self.collected_data['robot_state']]))
                        else:
                            part_group.create_dataset('qvel', data=np.array([getattr(state, f'{part}_vel') for state in self.collected_data['robot_state']]))
                            part_group.create_dataset('torque', data=np.array([getattr(state, f'{part}_torque') for state in self.collected_data['robot_state']]))
                            part_group.create_dataset('current', data=np.array([getattr(state, f'{part}_current') for state in self.collected_data['robot_state']]))
                            part_group.create_dataset('target_qvel', data=np.array([getattr(state, f'{part}_target_qvel') for state in self.collected_data['robot_state']]))
                            
                else:
                    self.get_logger().warn('No robot state data to save')
                    
                self.get_logger().info(f'HDF5 file saved to {h5_path}')
        except Exception as e:
            self.get_logger().error(f'Error saving HDF5 file: {str(e)}')

    def __del__(self):
        """Cleanup when the node is destroyed"""
        # Stop rate calculation thread
        self.rate_calculation_running = False
        if hasattr(self, 'rate_calculation_thread') and self.rate_calculation_thread.is_alive():
            self.rate_calculation_thread.join(timeout=1.0)
        
        # Stop keyboard listener
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()

def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Data Collection Node')
    parser.add_argument('--taskName', type=str, required=True,
                      help='Name of the task (default: test)')
    parser.add_argument('--horizon', type=int, default=1000,
                      help='Maximum number of steps per episode (default: 1000)')

    parsed_args = parser.parse_args(args=args)
    
    try:
        realname, row, col = parsed_args.taskName.split('_')
        for _ in range(5):
            print(f"\033[31m realname: {realname}, row: {row}, col: {col}\033[0m")
    except ValueError:
        print("wrong format")

    rclpy.init(args=args)
    node = DataCollectionNode(task_name=parsed_args.taskName, max_horizon=parsed_args.horizon)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
