#! /bin/zsh

if [ $# -ne 2 ]; then
    echo "Usage: $0 <dataset_path> <config_path>"
    exit 1
fi

dataset_path=$1
config_path=$2

sh scripts/view_aug.sh "$dataset_path"
python scripts/sample_meta.py --dataset_path "$dataset_path"
python nav2man/nav2man/scripts/train.py --config "$config_path"