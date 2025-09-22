# N2M Simulation Environment

![Simulation Experiment Video](docs/Feature4_robocasa_x3_3k.gif)

We provide the code to run N2M in robocasa environment. Note that this code is for `review purpose only` to show how the trained model works. We do not provide how to train the model for simulation environment.

## Installation
```
mamba create -n robocasa python=3.10
mamba activate robocasa

cd robocasa
pip install -e .

cd robomimic
pip install -e .

cd robosuite
pip install -e .

cd nav2man
pip install -e .

pip install -r requirements.txt

# Troubleshooting
from huggingface_hub import HfFolder, cached_download, hf_hub_download, model_info
=> from huggingface_hub import HfFolder, hf_hub_download, model_info

```

## Inference
Download trained weight from this <a href="">link</a> and place it under `models` folder. Then run the following command to see the results of our trained model.

```bash
# PnPCounterToCab
python ./scripts/1_data_collection_with_rollout.py --config "configs/robocasa/PnPCounterToCab.json" --sir_config "configs/n2m/PnPCounterToCab.json" --eval_only --SIR_sample_num 300 --robot_centric

# CloseDrawer
python ./scripts/1_data_collection_with_rollout.py --config "configs/robocasa/CloseDrawer.json" --sir_config "configs/n2m/CloseDrawer.json" --eval_only --SIR_sample_num 300 --robot_centric

# CloseDoubleDoor
python ./scripts/1_data_collection_with_rollout.py --config "configs/robocasa/CloseDoubleDoor.json" --sir_config "configs/n2m/CloseDoubleDoor.json" --eval_only --SIR_sample_num 300 --robot_centric

# OpenSingleDoor
python ./scripts/1_data_collection_with_rollout.py --config "configs/robocasa/OpenSingleDoor.json" --sir_config "configs/n2m/OpenSingleDoor.json" --eval_only --SIR_sample_num 300 --robot_centric
```