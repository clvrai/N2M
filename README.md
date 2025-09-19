# N2M Real-World Robotics System

This repository contains the complete codebase for N2M (Navigation to Manipulation) real-world experiments. The system is built on ROS2 framework for distributed robotics applications.

[Watch our system demonstration video](doc/Teaser_real.mov)


## System Architecture

Our system consists of a distributed architecture with two main components:

### Hardware Configuration
- **Desktop PC**: High-performance workstation for computation-intensive tasks
- **Onboard Computer (Jetson AGX)**: Embedded system for real-time sensor data processing and robot control
- **Communication**: Ethernet-based networking using ROS2 multi-machine communication protocol

### Software Architecture
- **Onboard Computer**: 
  - Sensor data acquisition and preprocessing
  - Real-time execution of motion commands
  - Low-level robot control interfaces
- **Desktop PC**: 
  - N2M neural network training and inference
  - Navigation planning and path optimization
  - Manipulation policy learning and execution

## Prerequisites and Environment Setup

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS
- **ROS2 Distribution**: Humble Hawksbill
- **Python**: 3.10+

### ROS2 Installation
Ensure ROS2 Humble is properly installed on both Desktop PC and Onboard Computer. Configure your shell environment by adding ROS2 setup to your `~/.zshrc`:

```bash
source /opt/ros/humble/setup.zsh
```

### Dependencies

#### Python Environment
We use `mamba` for Python environment management. If you don't have `mamba` installed, `conda` can be used as an alternative (simply replace `mamba` with `conda` in the commands below).

```bash
# Create and activate Python environment
mamba create -n rainbow python=3.10
mamba activate rainbow

# Install Python dependencies
pip install -r requirements.txt
```

#### ROS2 Workspace Setup

```bash
# Clone the repository with submodules
git clone --single-branch --branch real --recurse-submodules https://github.com/clvrai/N2M.git

# Build ROS2 packages for Desktop PC
colcon build --symlink-install \
  --packages-select realsense2_camera_msgs realsense2_camera realsense2_description \
                    rby_wrapper lakibeam1 zed_ros2 zed_components zed_wrapper \
  --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=YES

# Build ROS2 packages for Onboard PC (Jetson AGX)
colcon build --symlink-install \
  --packages-select realsense2_camera_msgs realsense2_camera realsense2_description \
                    rby_wrapper lakibeam1 \
  --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=YES
```
## Workflow

### Step 1: Expert Demonstration Data Collection

This step collects expert demonstration data for manipulation policy training through teleoperation.

#### Onboard Computer (Jetson AGX)
```bash
cd workbench/real

# Launch camera system
zsh launch_cam.sh

# Start teleoperation server
source install/setup.zsh
taskset -c 0,1,2,3 ros2 run rby_wrapper server_teleoperation.py
```

#### Desktop PC (Client)
```bash
# Source ROS2 environment
source install/setup.zsh

# Start data collection client (replace 'lamp' with your task name)
ros2 run rby_wrapper client_data_collection.py --taskName lamp

# Visualize collected data (specify HDF5 file or dataset path)
python utils/data_visualizer.py --hdf5_path <path>

# Verify video frame information
mediainfo --Full video_right.mp4 | grep -i frame
h5dump -H data.h5
```

### Step 2: Rollout Data Collection

This step collects rollout trajectories for N2M training by running the robot in various scenarios.

#### Onboard Computer (Jetson AGX)
```bash
# Launch camera system
zsh launch_cam.sh

# Start LiDAR system
source install/setup.zsh
ros2 launch lakibeam1 lakibeam1_scan_dual_lidar.launch.py

# Start robot server
source install/setup.zsh
taskset -c 0,1,2 ros2 run rby_wrapper server_rby1_robot.py
```

#### Desktop PC (Client)
```bash
# Data collection for N2M training
zsh n2m_manager.sh

# N2M inference with SIR (Sequential Importance Resampling) prediction
zsh n2m_manager.sh --inference

# Run manipulation policy
mamba activate craft
zsh run_client_manipulation_policy.sh
```

### Step 3: N2M Inference and Deployment

This step runs the trained N2M model for real-time navigation and manipulation tasks.

#### Onboard Computer (Jetson AGX)
```bash
# Launch camera system
zsh launch_cam.sh

# Start LiDAR system
source install/setup.zsh
ros2 launch lakibeam1 lakibeam1_scan_dual_lidar.launch.py

# Start robot server
source install/setup.zsh
taskset -c 0,1,2 ros2 run rby_wrapper server_rby1_robot.py
```

#### Desktop PC (Client)
```bash
# Data collection for N2M
zsh n2m_manager.sh

# N2M inference with SIR prediction
zsh n2m_manager.sh --inference

# Execute manipulation policy
mamba activate craft
zsh run_client_manipulation_policy.sh
```
<!-- # Acknowledgement -->


<!-- 
# Utils
```zsh
# zed camera, zed helper
zsh launch_zed.sh

# stitch
ros2 run rby_wrapper client_helper_stitch

# mapping
ros2 launch rby_wrapper mapping.launch.py

# lidar process
ros2 run rby_wrapper laser_helper

# vis
ros2 launch rby_wrapper display.launch.py

# base keyboard control
python utils/base_controller.py

# cam PCL
ros2 launch rby_wrapper campcl_helper.launch.py

# launch other cam
ros2 launch realsense2_camera rby_head_cam_depth.py

ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "name: 
    data: './maps/room1'"

# localization
ros2 launch rby_wrapper localization.launch.py map_file_name:=./maps/room
``` -->

