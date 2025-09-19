# Depandencies
```zsh
# For new python env
pip install -r ENV_FOR_ROS2.txt

# realsense
https://github.com/IntelRealSense/realsense-ros

# zed
https://github.com/stereolabs/zed-ros2-wrapper

# rby1 sdk
pip install rby1_sdk

# misc
pip install h5py keyboard
```

# sync
```zsh
sh sync_to_rainbow.sh
```

# complie
```zsh
# client
colcon build --symlink-install --packages-select realsense2_camera_msgs realsense2_camera realsense2_description rby_wrapper lakibeam1 zed_ros2 zed_components zed_wrapper --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=YES

# server
colcon build --symlink-install --packages-select realsense2_camera_msgs realsense2_camera realsense2_description rby_wrapper lakibeam1 --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=YES
```

# DATA collection
## Server (Jetson AGX)
```zsh
cd workbench/real

# cam
zsh launch_cam.sh

# teleoperation
source install/setup.zsh
taskset -c 0,1,2,3 ros2 run rby_wrapper server_teleoperation.py
```
## Client (User DESKTOP)
```zsh
source install/setup.zsh
ros2 run rby_wrapper client_data_collection.py --taskName lamp

# vis (hdf5 file path or whole dataset path)
python utils/data_visualizer.py --hdf5_path <path>

# check frame
mediainfo --Full video_right.mp4 | grep -i frame
h5dump -H data.h5
```

# Inference when Rollout
## Server (Jetson AGX)
```zsh
# launch cam
zsh launch_cam.sh

# Lidar
source install/setup.zsh
ros2 launch lakibeam1 lakibeam1_scan_dual_lidar.launch.py

# rby1 robot server
source install/setup.zsh
taskset -c 0,1,2 ros2 run rby_wrapper server_rby1_robot.py
```
## Client (User DESKTOP)
```zsh
# data collection for n2m
zsh n2m_manager.sh
# n2m inference with sir prediction
zsh n2m_manager.sh --inference

# policy
mamba activate craft
zsh run_client_manipulation_policy.sh
```

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
```
