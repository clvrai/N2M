#! /bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <sir_train_config_path>"
    exit 1
fi

sir_train_config_path=$1

python nav2man/nav2man/scripts/train.py --config "$sir_train_config_path"