#!/bin/zsh

# Exit on error
set -e

# Define color codes
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Source workspace setup
source install/setup.zsh

# # Launch ZED camera
# echo -e "${GREEN}Launching ZED camera...${NC}"
# ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i &

# Launch head camera
echo -e "${GREEN}Launching head camera...${NC}"
taskset -c 4,5 ros2 launch realsense2_camera rby_head_cam_img.py &

sleep 10

# Launch right camera
echo -e "${GREEN}Launching right camera...${NC}"
taskset -c 6,7 ros2 launch realsense2_camera rby_right_cam_img.py &

# sleep 5

# # Launch left camera
# echo -e "${GREEN}Launching left camera...${NC}"
# ros2 launch realsense2_camera rby_left_cam_img.py &

# Wait for all background processes
wait 