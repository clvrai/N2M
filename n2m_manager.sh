#!/bin/zsh

# Exit on error
set -e

# Define color codes
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Default values for flags
RUN_N2M=false

# Parse arguments
for arg in "$@"; do
  case $arg in
    --inference)
      RUN_N2M=true
      shift
      ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

# Source workspace setup
source install/setup.zsh

# Launch mapping
echo -e "${GREEN}Launching mapping...${NC}"
ros2 launch rby_wrapper mapping.launch.py &
sleep 1

# laser helper
echo -e "${GREEN}Launching laser helper...${NC}"
ros2 run rby_wrapper laser_helper &

sleep 1


# Launch rollout manager
echo -e "${GREEN}Launching rollout manager...${NC}"
ros2 run rby_wrapper client_rollout_manager.py &
sleep 1

if [ "$RUN_N2M" = true ]; then
  # SIR inference
  # echo -e "${GREEN}Launching SIR inference...${NC}"
  # ros2 run rby_wrapper client_SIR_inference.py&
  sleep 1


  echo -e "${GREEN}Launching client_naive_navigation...${NC}"
  ros2 run rby_wrapper client_naive_navigation.py &
  sleep 1

  # Launch pcl render
  echo -e "${GREEN}Launching head camera...${NC}"
  ros2 launch rby_wrapper cam_render.launch.py &

  sleep 1

  # Launch display (multi-robot)
  echo -e "${GREEN}Launching display...${NC}"
  ros2 launch rby_wrapper multi_robot_display.launch.py &

  sleep 1
  # use base_se2_helper to help transform predicted robot2 from base to map frame.
  echo -e "${GREEN}Launching base_se2_helper...${NC}"
  ros2 run rby_wrapper base_se2_helper &
  sleep 1

else
  # Launch display 
  echo -e "${GREEN}Launching display...${NC}"
  ros2 launch rby_wrapper display.launch.py &
fi

sleep 1

# stitch helper  
echo -e "${GREEN}Launching stitch helper...${NC}"
ros2 run rby_wrapper client_helper_stitch &

sleep 1

# zed helper 
echo -e "${GREEN}Launching client_pcl_frameID_converter...${NC}"
ros2 run rby_wrapper client_pcl_frameID_converter.py &

sleep 1

# Launch head camera
echo -e "${GREEN}Launching zed camera...${NC}"
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i &

# Wait for all background processes
wait 