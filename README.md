# N2M Simulation Environment

![Simulation Experiment Video](docs/Feature4_robocasa_x3_3k.gif)

We provide the code to run N2M in robocasa environment. Note that this code is for `review purpose only` to show how the trained model works. We do not provide how to train the model for simulation environment.

## Installation

## Inference
Download trained weight from this <a href="">link</a> and place it under `models` folder. Then run the following command to see the results of our trained model.

```bash
# PnPCounterToCab
python ./scripts/1_data_collection_with_rollout.py --config "configs/robocasa/PnPCounterToCab.json" --sir_config "configs/n2m/PnPCounterToCab.json" --eval_only --SIR_sample_num 300 --robot_centric

# CloseDrawer
python ./scripts/1_data_collection_with_rollout.py --config "configs/robocasa/CloseDrawer.json" --sir_config "configs/n2m/CloseDrawer.json" --eval_only --SIR_sample_num 300 --robot_centric

# CloseDoubleDoor
python ./scripts/1_data_collection_with_rollout.py --config "configs/robocasa/CloseDrawer.json" --sir_config "configs/n2m/PnPCounCloseDrawerterToCab.json" --eval_only --SIR_sample_num 300 --robot_centric

# OpenSingleDoor
python ./scripts/1_data_collection_with_rollout.py --config "configs/robocasa/CloseDrawer.json" --sir_config "configs/n2m/CloseDrawer.json" --eval_only --SIR_sample_num 300 --robot_centric
```