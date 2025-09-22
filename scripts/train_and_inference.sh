#! /bin/bash

if [ $# -lt 3 ]; then
    echo "Usage: $0 <sir_train_config_path> <sir_inference_config_path> <rollout_config_path> <robot_centric>"
    exit 1
fi

sir_train_config_path=$1
sir_inference_config_path=$2
rollout_config_path=$3
robot_centric=$4

python nav2man/nav2man/scripts/train.py --config "$sir_train_config_path"
if [ -n "$robot_centric" ]; then
    python ./scripts/1_data_collection_with_rollout.py --config "$rollout_config_path" --sir_config "$sir_inference_config_path" --eval_only --SIR_sample_num 300 --robot_centric
else
    python ./scripts/1_data_collection_with_rollout.py --config "$rollout_config_path" --sir_config "$sir_inference_config_path" --eval_only --SIR_sample_num 300
fi