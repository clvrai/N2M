#!/bin/zsh

# Exit on error
set -e

# Define color codes
GREEN='\033[0;32m'
NC='\033[0m' # No Color


# Source workspace setup
source install/setup.zsh

echo -e "${GREEN}Launching data collection...${NC}"
ros2 run rby_wrapper client_data_collection.py --taskName lamp0620_2_3 &

sleep 1

echo -e "${GREEN}Launching base controller...${NC}"
python utils/base_controller.py &

wait 