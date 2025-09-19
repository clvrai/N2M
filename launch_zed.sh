#!/bin/zsh

# Exit on error
set -e

# Define color codes
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Source workspace setup
source install/setup.zsh

# Launch head camera
echo -e "${GREEN}Launching zed camera...${NC}"
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i &

sleep 5

# zed helper
echo -e "${GREEN}Launching zed camera...${NC}"
ros2 run rby_wrapper client_pcl_frameID_converter.py &

# Wait for all background processes
wait 