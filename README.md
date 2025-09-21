# N2M Real-World Robotics System

This repository contains the full implementation of our real-world experiments. We use ROS2(Humble) to organize the whole system.

<p align="center">
  <img src="doc/Teaser_real.gif" alt="System Demonstration">
  <br>
  <em>Real-world experiments in N2M</em>
</p>

## System Architecture

Our system consists of a distributed architecture with two main components:

### Hardware
- **Desktop**: High-performance workstation for computation-intensive tasks.
- **Onboard (Jetson AGX)**: Embedded system for real-time sensor data processing and robot control.
- **Communication**: Ethernet-based networking using ROS2 multi-machine communication protocol.
- **Mobile Manipulator**: Rainbow Robotics RB-Y1. Check [manual](https://rainbowrobotics.github.io/rby1-dev/) for more info.

### Software
- **Onboard (Jetson AGX)**: 
  - Sensor data acquisition and preprocessing
  - Real-time execution of motion commands
  - Low-level robot control interfaces
- **Desktop (Ubuntu 22.04 LTS)**: 
  - N2M neural network training and inference
  - Navigation planning and path optimization
  - Manipulation policy learning and execution

## Prerequisites and Environment Setup

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS
- **ROS2 Distribution**: 
- **Python**: 3.10

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

<p align="center">
  <img src="doc/UI_expert_data_collection.png" 
       alt="System Demonstration"
       width="80%">
  <br>
  <em>User Inferface for Collecting Expert Demonstrations</em>
</p>

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
cd workbench/real

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

<p align="center">
  <img src="doc/tf_tree_rollout.jpg" 
       alt="System Demonstration"
       width="50%">
  <br>
  <em>TF tree during N2M rollout</em>
</p>

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
zsh n2m_manager.sh

# Run manipulation policy
mamba activate craft
zsh run_client_manipulation_policy.sh
```

### Step 3: N2M Inference and Deployment
This step runs the trained N2M model for real-time navigation and manipulation tasks.

<p align="center">
  <img src="doc/N2M_inference.png" 
       alt="System Demonstration"
       width="80%">
  <br>
  <em>Rosgraph during N2M inference</em>
</p>


<p align="center">
  <img src="doc/tf_tree_inference.jpg" 
       alt="System Demonstration"
       width="80%">
  <br>
  <em>TF tree during N2M inference</em>
</p>


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

## User Interface
We developed a simple UI that publishes specific ROS2 topics through keyboard inputs, enabling robot state transitions via callbacks to achieve specific functionalities. This interface program is invoked when executing the `n2m_manager.sh` script during rollout collection and N2M inference. The UI is shown in the figure below:
<p align="center">
  <img src="doc/UI_rollout_collection_N2M_inference.png" 
       alt="System Demonstration"
       width="80%">
  <br>
  <em>User Inferface for Rollout Collection and N2M Inference</em>
</p>

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

