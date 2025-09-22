# Set the camera for each scene
```



```


# collision-free randomization
1. should create occ for each scene
2. implement collision-free sampling




# Downloading dataset
```zsh
# Play back sample demonstrations of tasks (to give a overall understanding of a task)
python -m robocasa.demos.demo_tasks

# ds_types: mg_im for training, human_raw for downloading test
# task: PnPCounterToCab, CloseDoubleDoor, OpenSingleDoor, CloseDrawer
python robocasa/robocasa/scripts/download_datasets.py --ds_types mg_im --task CloseDoubleDoor
```
# dataset info
```zsh
python robomimic/robomimic/scripts/get_dataset_info.py --dataset <TBD>
```

# Training Manipulation policy
```zsh
CUDA_VISIBLE_DEVICES=<GPU-id> 
# pnpCounterToCab_BCtransformer
python ./robomimic/robomimic/scripts/train.py --config models/pnpCounterToCab/BCtransformer/pnpCounterToCab_BCtransformer.json
# PnPCounterToCab_diffusion
python ./robomimic/robomimic/scripts/train.py --config models/pnpCounterToCab/diffusion/PnPCounterToCab_diffusion.json



# CloseDoubleDoor_BCtransformer
python ./robomimic/robomimic/scripts/train.py --config models/CloseDoubleDoor/BCtransformer/CloseDoubleDoor_BCtransformer.json
# CloseDoubleDoor_diffusion
python ./robomimic/robomimic/scripts/train.py --config models/CloseDoubleDoor/diffusion/CloseDoubleDoor_diffusion.json
# CloseDoubleDoor_iql
python ./robomimic/robomimic/scripts/train.py --config models/CloseDoubleDoor/iql/CloseDoubleDoor_iql.json

# OpenSingleDoor_BCtransformer
python ./robomimic/robomimic/scripts/train.py --config models/OpenSingleDoor/BCtransformer/OpenSingleDoor_BCtransformer.json
# OpenSingleDoor_diffusion
python ./robomimic/robomimic/scripts/train.py --config models/OpenSingleDoor/diffusion/OpenSingleDoor_diffusion.json

# CloseDrawer_BCtransformer
python ./robomimic/robomimic/scripts/train.py --config models/CloseDrawer/BCtransformer/CloseDrawer_BCtransformer.json
# CloseDrawer_diffusion
python ./robomimic/robomimic/scripts/train.py --config models/CloseDrawer/diffusion/CloseDrawer_diffusion.json

```

# Simulation
```
1. set robot pos + forward [done]
2. set cameras for each scene (will be very hard and time consuming)
3. set real navigation for each scene (with pre-build map)
4. collect data with full scene.
```
## rollout for data collection
```zsh
# fixed
CUDA_VISIBLE_DEVICES=<gpu-id> python ./scripts/1_data_collection_with_rollout.py --config <config_path> --eval_only

# randomize for SIR
CUDA_VISIBLE_DEVICES=<gpu-id> python ./scripts/1_data_collection_with_rollout.py --config <config_path> --eval_only --randomize_base

# randomize for inference
CUDA_VISIBLE_DEVICES=<gpu-id> python ./scripts/1_data_collection_with_rollout.py --config <config_path> --eval_only --randomize_base --inference_mode

# test with fixed base position
CUDA_VISIBLE_DEVICES=<gpu-id> python ./scripts/1_data_collection_with_rollout.py --config ./models/pnpCounterToCab/BCtransformer/pnpCounterToCab_BCtransformer_rollout_<specify name>.json --eval_only

# test with randomized base position
CUDA_VISIBLE_DEVICES=<gpu-id> python ./scripts/1_data_collection_with_rollout.py --config ./models/pnpCounterToCab/BCtransformer/pnpCounterToCab_BCtransformer_rollout_<specify name>.json --eval_only --randomize_base

# test with SIR
CUDA_VISIBLE_DEVICES=<gpu-id> python ./scripts/1_data_collection_with_rollout.py --config ./models/pnpCounterToCab/BCtransformer/pnpCounterToCab_BCtransformer_rollout_<specify name>.json --sir_config models/pnpCounterToCab/BCtransformer/pnpCounterToCab_BCtransformer_SIR.json --eval_only
```

## view point augmentation
```zsh
# sample viewpoints
python nav2man/nav2man/scripts/sample_camera_poses.py --dataset_path <dataset_path> --num_poses 300

# generate augmented views
nav2man/nav2man/scripts/render/build/fpv_render <dataset_path>

# do it together
sh scripts/view_aug.sh <dataset_path>
```

## Robot Centric view sampling
```zsh
python nav2man/nav2man/scripts/get_robot_centric_camera_extrinsics.py --dataset_path <dataset_path>

nav2man/nav2man/scripts/render_robot_centric_viewpoint/build/fpv_render <dataset_path>
```

## SIR training
IMPORTANT: RENAME ROLLOUTS DIR NAME
```zsh
CUDA_VISIBLE_DEVICES=<gpu-id> python nav2man/nav2man/scripts/train.py --config <config_path>
```

## View Point augmentation + training
```zsh
CUDA_VISIBLE_DEVICES=<gpu-id> sh scripts/train_SIR.sh <dataset_path> <config_path>
```

## Down Sample Rollouts
```zsh
python scripts/sample_meta.py --dataset_path <dataset_path>
```

## Inference
```zsh
CUDA_VISIBLE_DEVICES=<gpu-id> python ./scripts/1_data_collection_with_rollout.py --config <inference_config> --sir_config <sir_inference_config> --eval_only

CUDA_VISIBLE_DEVICES=<gpu-id> python ./scripts/1_data_collection_with_rollout.py --config <inference_config> --sir_config <sir_inference_config> --eval_only --SIR_sample_num 300

CUDA_VISIBLE_DEVICES=<gpu-id> python ./scripts/1_data_collection_with_rollout.py --config <inference_config> --sir_config <sir_inference_config> --eval_only --SIR_sample_num 300 --robot_centric
```

## Train and Inference
```zsh
CUDA_VISIBLE_DEVICES=<gpu-id> sh scripts/train_and_inference.sh <sir_train_config_path> <sir_inference_config_path> <rollout_config_path> <robot_centric>
```

# real-world ()

## Training: train the proposed model based on rollout data


## Inference: estimate suitable interaction region (SIR)


# Error shooting
```zsh
# ImportError: Failed to load GLFW3 shared library
# This bug is caused by VSCode's integrated terminal. It is recommended to run the code in an independent terminal.
sudo apt install mlocate
locate libglfw.so.3
sudo cp /usr/lib/x86_64-linux-gnu/libglfw.so.3 /usr/local/lib

# AttributeError: module 'torch' has no attribute 'compiler'
# This is caused by torch version. install 2.2.1 (robomimic may require 2.0.1, but due to backward compatability, it is fine.)
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1

# Failed to import transformers.generation.utils
# This is caused by transformer version. should reinstall
pip install "transformers[sklearn]" --force-reinstall

# ImportError: cannot import name 'cached_download' from 'huggingface_hub'
# This is caused by the package itself. 
# remove cached_download from the import line in the file /home/xx/miniconda3/envs/robocasa/lib/python3.10/site-packages/diffusers/dynamic_modules_utils.py
# https://github.com/easydiffusion/easydiffusion/issues/1851
from huggingface_hub import HfFolder, cached_download, hf_hub_download, model_info
=> from huggingface_hub import HfFolder, hf_hub_download, model_info

# no moduel named 'open3d'
pip install open3d

# numtpy version error. (due to the confiction of dependencies, before you run robomimic train.py, you should install robomimic to satisfy its dependencies.)
cd robomimic
pip install -e .

```


# syncing the ckpts
```zsh
rsync -avzh --progress /data/hyunjun/robocasa/datasets/rollouts/CloseDrawer_BCtransformer_rollouts/CloseDrawer_BCtransformer_rollout_scene1/rollouts_sir/SIR kaist2:/data/kaixin_data/robocasa/datasets/rollouts/CloseDrawer_BCtransformer_rollouts/CloseDrawer_BCtransformer_rollout_scene1/rollouts_sir
```