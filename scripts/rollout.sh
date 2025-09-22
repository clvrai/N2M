#! /bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <rollout_config_path> <inference_mode>"
    exit 1
fi

rollout_config_path=$1
inference_mode=$2

if [ -n "$inference_mode" ]; then
    python ./scripts/1_data_collection_with_rollout.py --config "$rollout_config_path" --eval_only --randomize_base --inference_mode
else
    python ./scripts/1_data_collection_with_rollout.py --config "$rollout_config_path" --eval_only --randomize_base
fi