# N2M: Bridging Navigation and Manipulation by Learning Pose Preference from Rollout
Kaixin Chai*, Hyunjun Lee*, Joseph J. Lim

![System Overview](doc/System_Overview.png)

This is an official implementation of N2M. `main` branch includes the use of n2m module only. For examples use cases in simulation and real world, please refer to `sim` and `real` branches respectively.

## TODO
- Organize README
- Check installation
- Check Training

## Installation
Clone and install necessary packages
```
git clone --single-branch --branch main https://github.com/clvrai/N2M.git
cd N2M

mamba create -n n2m python==3.11
pip install -r requirements.txt
pip install -e .
```

## Training
We provided detailed instructions to train N2M module.

### Data preparation

![Data Preparation](doc/Data_Preparation.png)

You should first prepare raw data with pairs of local scene and preferable initial pose. Local scene is a point cloud of a scene and you may stitch point clouds using multiple calibrated cameras. In this repo, we do not provide code for capturing the local scene.

The format of raw data should be placed under `dataset` folder in the format below
```

```