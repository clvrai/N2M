#!/bin/zsh

# Exit on error
set -e

# Define color codes
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Source workspace setup
source install/setup.zsh

export PYTHONPATH=/home/mm/miniconda3/envs/craft/lib/python3.10/site-packages:$PYTHONPATH
echo "PYTHONPATH set to prioritize conda environment"

ros2 run rby_wrapper client_manipulation_dp.py

wait 