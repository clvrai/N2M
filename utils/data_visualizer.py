#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.gridspec import GridSpec
import glob

class HDF5Visualizer:
    def __init__(self, hdf5_path, need_vis_left=False, show_plot=True, show_normalized=True):
        """Initialize the HDF5 visualizer with the path to the HDF5 file."""
        self.hdf5_path = hdf5_path
        self.need_vis_left = need_vis_left
        self.show_plot = show_plot
        self.show_normalized = show_normalized
        self.data = {}
        self.time_steps = None
        
        # Only include parts we want to visualize
        if need_vis_left:
            self.robot_parts = ['left_arm', 'right_arm', 'left_gripper', 'right_gripper']
        else:
            self.robot_parts = ['right_arm', 'right_gripper']
            
        # We only care about position data
        self.data_types = ['qpos', 'target_qpos']
        if show_normalized:
            self.data_types.extend(['qpos_normalized', 'target_qpos_normalized'])
        self.gripper_parts = ['left_gripper', 'right_gripper']
        
    def load_data(self):
        """Load data from the HDF5 file."""
        print(f"Loading data from {self.hdf5_path}...")
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load all robot part data
            for part in self.robot_parts:
                if part in f:
                    self.data[part] = {}
                    for data_type in self.data_types:
                        if data_type in f[part]:
                            self.data[part][data_type] = f[part][data_type][:]
            
            # Determine the number of time steps
            if len(self.data) > 0:
                first_part = list(self.data.keys())[0]
                first_data_type = list(self.data[first_part].keys())[0]
                self.time_steps = self.data[first_part][first_data_type].shape[0]
                print(f"Data loaded. Found {self.time_steps} time steps.")
            else:
                print("No data found in the HDF5 file.")
    
    def create_time_axis(self):
        """Create a time axis for plotting."""
        if self.time_steps is None:
            return None
        return np.arange(self.time_steps)
    
    def visualize_position_vs_target(self):
        """Visualize actual positions vs target positions for each joint."""
        if not self.data:
            print("No data to visualize. Please load data first.")
            return
        
        time_axis = self.create_time_axis()
        if time_axis is None:
            print("Could not create time axis.")
            return
        
        # Create a large figure
        plt.figure(figsize=(20, 24))
        plt.suptitle("Position vs Target Position", fontsize=24)
        
        # Count total number of parts that have both qpos and target_qpos
        plot_count = 0
        for part in self.robot_parts:
            if part in self.data and 'qpos' in self.data[part] and 'target_qpos' in self.data[part]:
                plot_count += 1
                
                # Add additional plot for normalized data if it's a gripper and we should show normalized
                if part in self.gripper_parts and self.show_normalized and 'qpos_normalized' in self.data[part]:
                    plot_count += 1
        
        if plot_count == 0:
            print("No position and target position data found to compare.")
            return
        
        grid = GridSpec(plot_count, 1, height_ratios=[1] * plot_count)
        
        idx = 0
        for part in self.robot_parts:
            if part not in self.data or 'qpos' not in self.data[part] or 'target_qpos' not in self.data[part]:
                continue
            
            # Plot raw position data
            ax = plt.subplot(grid[idx])
            idx += 1
            ax.set_title(f"{part.replace('_', ' ').title()} - Position vs Target", fontsize=18)
            
            qpos = self.data[part]['qpos']
            target_qpos = self.data[part]['target_qpos']
            
            # Handle different data shapes
            if len(qpos.shape) == 1:  # Scalar values (e.g., gripper position)
                ax.plot(time_axis, qpos, label="Actual Position", color='blue')
                ax.plot(time_axis, target_qpos, label="Target Position", color='red', linestyle='--')
            else:  # Array values (e.g., arm joint positions)
                for j in range(qpos.shape[1]):
                    # Use different line styles for actual and target
                    ax.plot(time_axis, qpos[:, j], label=f"Actual[{j}]" if j == 0 else "_nolegend_", color=f'C{j}')
                    ax.plot(time_axis, target_qpos[:, j], label=f"Target[{j}]" if j == 0 else "_nolegend_", 
                           color=f'C{j}', linestyle='--')
            
            ax.set_xlabel("Time Steps", fontsize=12)
            ax.set_ylabel("Position", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=10)
            
            # If this is a gripper and we should show normalized data, add a normalized plot
            if part in self.gripper_parts and self.show_normalized and 'qpos_normalized' in self.data[part] and 'target_qpos_normalized' in self.data[part]:
                ax = plt.subplot(grid[idx])
                idx += 1
                ax.set_title(f"{part.replace('_', ' ').title()} - Normalized Position vs Target", fontsize=18)
                
                qpos_norm = self.data[part]['qpos_normalized']
                target_qpos_norm = self.data[part]['target_qpos_normalized']
                
                # Handle different data shapes
                if len(qpos_norm.shape) == 1:  # Scalar values
                    ax.plot(time_axis, qpos_norm, label="Actual Normalized", color='blue')
                    ax.plot(time_axis, target_qpos_norm, label="Target Normalized", color='red', linestyle='--')
                else:  # Array values
                    for j in range(qpos_norm.shape[1]):
                        ax.plot(time_axis, qpos_norm[:, j], label=f"Actual Norm[{j}]" if j == 0 else "_nolegend_", color=f'C{j}')
                        ax.plot(time_axis, target_qpos_norm[:, j], label=f"Target Norm[{j}]" if j == 0 else "_nolegend_", 
                               color=f'C{j}', linestyle='--')
                
                ax.set_xlabel("Time Steps", fontsize=12)
                ax.set_ylabel("Normalized Position [0-1]", fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=10)
                
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
                
                ax.set_ylim([-0.1, 1.1])
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for title
        plt.subplots_adjust(hspace=0.4)
        
        # Save figure
        output_dir = os.path.dirname(self.hdf5_path)
        output_filename = f"position_vs_target_{os.path.basename(os.path.dirname(self.hdf5_path))}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=150)
        print(f"Position vs target visualization saved to {output_path}")
        
        # Show figure only if required
        if self.show_plot:
            plt.show()
        else:
            plt.close()
    
    def run_visualization(self):
        """Run the visualization process."""
        self.load_data()
        if self.data:
            self.visualize_position_vs_target()

def find_hdf5_files(directory_path):
    """Find all HDF5 files in the given directory structure."""
    # Check if path is a direct file
    if os.path.isfile(directory_path) and (directory_path.endswith('.h5') or directory_path.endswith('.hdf5')):
        return [directory_path]
    
    # Check for episode subdirectories
    hdf5_files = []
    
    # Look for numbered subdirectories
    subdirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(directory_path, subdir)
        
        # Look for data.h5 or data.hdf5 in this subdirectory
        for ext in ['.h5', '.hdf5']:
            h5_path = os.path.join(subdir_path, f'data{ext}')
            if os.path.exists(h5_path):
                hdf5_files.append(h5_path)
                break
    
    return hdf5_files

def main():
    parser = argparse.ArgumentParser(description='Visualize robot data from HDF5 file')
    parser.add_argument('--hdf5_path', type=str, default='/home/mm/workbench/dataset/lamp/1412/data.h5',
                        help='Path to the HDF5 file or directory containing HDF5 files (default: dataset/test/0/data.h5)')
    parser.add_argument('--need_vis_left', action='store_true', default=False,
                        help='Whether to visualize left arm and left gripper (default: False)')
    parser.add_argument('--no_normalized', action='store_true', default=False,
                        help='Disable visualization of normalized gripper data (default: False)')
    args = parser.parse_args()
    
    # Resolve relative paths
    hdf5_path = os.path.abspath(args.hdf5_path)
    
    # Check if path exists
    if not os.path.exists(hdf5_path):
        print(f"Error: Path {hdf5_path} does not exist.")
        return
    
    # Find HDF5 files
    hdf5_files = find_hdf5_files(hdf5_path)
    
    if not hdf5_files:
        print(f"Error: No HDF5 files found at {hdf5_path}")
        return
    
    # Determine if we should show plots (only for single file)
    show_plot = len(hdf5_files) == 1
    
    # Process each file
    for i, file_path in enumerate(hdf5_files):
        print(f"Processing file {i+1}/{len(hdf5_files)}: {file_path}")
        
        # Create and run visualizer
        visualizer = HDF5Visualizer(file_path, 
                                   need_vis_left=args.need_vis_left, 
                                   show_plot=show_plot,
                                   show_normalized=not args.no_normalized)
        try:
            visualizer.run_visualization()
        except Exception as e:
            print(f"Error visualizing {file_path}: {e}")
    
    print(f"Visualization completed for {len(hdf5_files)} files.")

if __name__ == '__main__':
    main() 