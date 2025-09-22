#! /bin/zsh

if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset_path>"
    exit 1
fi

dataset_path=$1

python nav2man/nav2man/scripts/sample_camera_poses.py --dataset_path "$dataset_path" --num_poses 300
nav2man/nav2man/scripts/render/build/fpv_render "$dataset_path"