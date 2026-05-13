# N2M-Benchmark

Unified benchmark framework for mobile manipulation navigation predictors.

## Overview

This repository provides a benchmark environment for evaluating different navigation prediction methods (N2M, Mobipi, LeLaN, Blank, Reachability) combined with manipulation policies in RoboCasa kitchen environments.

**Key Features:**

- Unified predictor/policy/env interfaces for benchmark
- include 5 predictors: N2M, Mobipi, LeLaN, Blank, Reachability
- Minimal modification to third-party libraries
- Hydra configuration management

## Installation

```bash
mamba create -c conda-forge -n n2m_benchmark python=3.10 -y
mamba activate n2m_benchmark

git clone git@github.com:clvrai/N2M.git -b benchmark N2M_benchmark
cd N2M_benchmark

chmod +x install.sh
./install.sh    # it is ok to have pip packages conflicts.
```

## Prepare Policy

### Policy1: Robomimic

**Download datasets**
Download robocasa pre-collected dataset for policy training and inference

```bash
cd env/robocasa
python robocasa/scripts/my_download_datasets.py --ds_types mg_im --tasks PnPCounterToCab --download_dir ../../data/policy/robomimic/datasets
python robocasa/scripts/my_download_datasets.py --ds_types mg_im --tasks CloseDoubleDoor --download_dir ../../data/policy/robomimic/datasets
python robocasa/scripts/my_download_datasets.py --ds_types mg_im --tasks OpenSingleDoor --download_dir ../../data/policy/robomimic/datasets
python robocasa/scripts/my_download_datasets.py --ds_types mg_im --tasks CloseDrawer --download_dir ../../data/policy/robomimic/datasets
```

**Train policies**
You can train yourself, or use our pre-trained policy checkpoints. 

Click to expand training commands

```bash
cd policy/robomimic

# PnPCounterToCab - BC Transformer
python robomimic/scripts/train.py --config ../../data/policy/robomimic/configs/pnpCounterToCab_BCtransformer.json

# PnPCounterToCab - Diffusion
python robomimic/scripts/train.py --config ../../data/policy/robomimic/configs/PnPCounterToCab_diffusion.json

# CloseDoubleDoor - BC Transformer
python robomimic/scripts/train.py --config ../../data/policy/robomimic/configs/CloseDoubleDoor_BCtransformer.json

# CloseDoubleDoor - Diffusion
python robomimic/scripts/train.py --config ../../data/policy/robomimic/configs/CloseDoubleDoor_diffusion.json

# OpenSingleDoor - BC Transformer
python robomimic/scripts/train.py --config ../../data/policy/robomimic/configs/OpenSingleDoor_BCtransformer.json

# OpenSingleDoor - Diffusion
python robomimic/scripts/train.py --config ../../data/policy/robomimic/configs/OpenSingleDoor_diffusion.json

# CloseDrawer - BC Transformer
python robomimic/scripts/train.py --config ../../data/policy/robomimic/configs/CloseDrawer_BCtransformer.json

# CloseDrawer - Diffusion
python robomimic/scripts/train.py --config ../../data/policy/robomimic/configs/CloseDrawer_diffusion.json
```



Policy checkpoints should be placed as follows:

```
data/policy/robomimic/checkpoints/
├── pnpCounterToCab_BCtransformer.pth
├── PnPCounterToCab_diffusion.pth
├── CloseDoubleDoor_BCtransformer.pth
├── CloseDoubleDoor_diffusion.pth
├── OpenSingleDoor_BCtransformer.pth
├── OpenSingleDoor_diffusion.pth
├── CloseDrawer_BCtransformer.pth
└── CloseDrawer_diffusion.pth
```

### Policy2: VLM

Not implemented. 

### Policy3: VLA

Not implemented.

## Prepare Predictors

### Predictor1: N2M

Collect Policy rollout for N2M training

```bash
# name: [CloseDrawer, PnPCounterToCab, CloseDoubleDoor, OpenSingleDoor]
# policy: [bc_transformer, diffusion]
CUDA_VISIBLE_DEVICES=0 python scripts/collect_n2m_data.py \
  env.name=CloseDrawer \
  env.render=false \
  'env.layout_and_style_ids=[[0,0]]' \
  policy=diffusion \
  benchmark=collection \
  benchmark.num_valid_data=50

CUDA_VISIBLE_DEVICES=1 python scripts/collect_n2m_data.py \
  env.name=CloseDrawer \
  env.render=false \
  'env.layout_and_style_ids=[[0,0]]' \
  policy=bc_transformer \
  benchmark=collection \
  benchmark.num_valid_data=50
```

Output format:

```
data/predictor/n2m/{task}_{scene}_{style}_{policytype}/
├── pcl/
│   ├── 0.pcd
│   ├── 1.pcd
│   └── ...
└── meta.json  # camera info, pose with pcd_path
```

Data augmentation and Train N2M module

```bash
# put this into installation.sh
cd predictor/N2M/scripts/render
mkdir build && cd build
cmake .. && make -j && cd ../../../../..

python predictor/N2M/scripts/sample_camera_poses.py --dataset_path data/predictor/n2m/CloseDrawer_0_0_diffusion_20 --num_poses 300 --num_episodes 20
predictor/N2M/scripts/render/build/fpv_render data/predictor/n2m/CloseDrawer_0_0_diffusion_20 
CUDA_VISIBLE_DEVICES=0 python predictor/N2M/scripts/train.py --use_cache --max_epoch 1000 --num_gaussians 2 --dataset_path ./data/predictor/n2m/CloseDrawer_0_0_diffusion_20 --encoder_ckpt ./data/predictor/n2m/PointBERT/PointTransformer_ModelNet8192points.pth

python predictor/N2M/scripts/sample_camera_poses.py --dataset_path data/predictor/n2m/CloseDrawer_0_0_diffusion_35 --num_poses 300 --num_episodes 35
predictor/N2M/scripts/render/build/fpv_render data/predictor/n2m/CloseDrawer_0_0_diffusion_35 
CUDA_VISIBLE_DEVICES=1 python predictor/N2M/scripts/train.py --use_cache --max_epoch 1000 --num_gaussians 2 --dataset_path ./data/predictor/n2m/CloseDrawer_0_0_diffusion_35 --encoder_ckpt ./data/predictor/n2m/PointBERT/PointTransformer_ModelNet8192points.pth

python predictor/N2M/scripts/sample_camera_poses.py --dataset_path data/predictor/n2m/CloseDrawer_0_0_diffusion_50 --num_poses 300 --num_episodes 50
predictor/N2M/scripts/render/build/fpv_render data/predictor/n2m/CloseDrawer_0_0_diffusion_50 
CUDA_VISIBLE_DEVICES=3 python predictor/N2M/scripts/train.py --use_cache --max_epoch 1000 --num_gaussians 2 --dataset_path ./data/predictor/n2m/CloseDrawer_0_0_diffusion_50 --encoder_ckpt ./data/predictor/n2m/PointBERT/PointTransformer_ModelNet8192points.pth


python predictor/N2M/scripts/sample_camera_poses.py --dataset_path data/predictor/n2m/CloseDrawer_0_0_bc_transformer_20 --num_poses 300 --num_episodes 20
predictor/N2M/scripts/render/build/fpv_render data/predictor/n2m/CloseDrawer_0_0_bc_transformer_20 
CUDA_VISIBLE_DEVICES=5 python predictor/N2M/scripts/train.py --use_cache --max_epoch 1000 --num_gaussians 2 --dataset_path ./data/predictor/n2m/CloseDrawer_0_0_bc_transformer_20 --encoder_ckpt ./data/predictor/n2m/PointBERT/PointTransformer_ModelNet8192points.pth

python predictor/N2M/scripts/sample_camera_poses.py --dataset_path data/predictor/n2m/CloseDrawer_0_0_bc_transformer_35 --num_poses 300 --num_episodes 35
predictor/N2M/scripts/render/build/fpv_render data/predictor/n2m/CloseDrawer_0_0_bc_transformer_35 
CUDA_VISIBLE_DEVICES=6 python predictor/N2M/scripts/train.py --use_cache --max_epoch 1000 --num_gaussians 2 --dataset_path ./data/predictor/n2m/CloseDrawer_0_0_bc_transformer_35 --encoder_ckpt ./data/predictor/n2m/PointBERT/PointTransformer_ModelNet8192points.pth

python predictor/N2M/scripts/sample_camera_poses.py --dataset_path data/predictor/n2m/CloseDrawer_0_0_bc_transformer_50 --num_poses 300 --num_episodes 50
predictor/N2M/scripts/render/build/fpv_render data/predictor/n2m/CloseDrawer_0_0_bc_transformer_50 64
CUDA_VISIBLE_DEVICES=7 python predictor/N2M/scripts/train.py --use_cache --max_epoch 1000 --num_gaussians 2 --dataset_path ./data/predictor/n2m/CloseDrawer_0_0_bc_transformer_50 --encoder_ckpt ./data/predictor/n2m/PointBERT/PointTransformer_ModelNet8192points.pth
```

### Predictor2: Mobipi

Reconstruct the scene in mobipi their own repo (6m57s for each reconstruction, including 83s for capture image and pcd. 3dgs with ground truth transform_matrix and pcd.)

```bash
CUDA_VISIBLE_DEVICES=7 python mobipi/scene_model/collect_images.py --env_name CloseDoubleDoor --layout_id 0 --style_id 1 --seed 123000
CUDA_VISIBLE_DEVICES=0 python mobipi/scene_model/collect_images.py --env_name CloseDoubleDoor --layout_id 0 --style_id 3 --seed 123000
CUDA_VISIBLE_DEVICES=1 python mobipi/scene_model/collect_images.py --env_name CloseDoubleDoor --layout_id 0 --style_id 4 --seed 123000
CUDA_VISIBLE_DEVICES=2 python mobipi/scene_model/collect_images.py --env_name CloseDoubleDoor --layout_id 0 --style_id 5 --seed 123000
CUDA_VISIBLE_DEVICES=3 python mobipi/scene_model/collect_images.py --env_name CloseDoubleDoor --layout_id 0 --style_id 7 --seed 123000
CUDA_VISIBLE_DEVICES=4 python mobipi/scene_model/collect_images.py --env_name CloseDoubleDoor --layout_id 0 --style_id 8 --seed 123000
CUDA_VISIBLE_DEVICES=5 python mobipi/scene_model/collect_images.py --env_name CloseDoubleDoor --layout_id 0 --style_id 9 --seed 123000
CUDA_VISIBLE_DEVICES=6 python mobipi/scene_model/collect_images.py --env_name PnPCounterToCab --layout_id 5 --style_id 6 --seed 123000
CUDA_VISIBLE_DEVICES=0 python mobipi/scene_model/collect_images.py --env_name OpenSingleDoor --layout_id 0 --style_id 0 --seed 123000
CUDA_VISIBLE_DEVICES=1 python mobipi/scene_model/collect_images.py --env_name OpenSingleDoor --layout_id 0 --style_id 1 --seed 123000
CUDA_VISIBLE_DEVICES=2 python mobipi/scene_model/collect_images.py --env_name OpenSingleDoor --layout_id 0 --style_id 2 --seed 123000
CUDA_VISIBLE_DEVICES=3 python mobipi/scene_model/collect_images.py --env_name OpenSingleDoor --layout_id 0 --style_id 3 --seed 123000
CUDA_VISIBLE_DEVICES=4 python mobipi/scene_model/collect_images.py --env_name OpenSingleDoor --layout_id 0 --style_id 4 --seed 123000
CUDA_VISIBLE_DEVICES=5 python mobipi/scene_model/collect_images.py --env_name OpenSingleDoor --layout_id 0 --style_id 5 --seed 123000
CUDA_VISIBLE_DEVICES=6 python mobipi/scene_model/collect_images.py --env_name OpenSingleDoor --layout_id 0 --style_id 6 --seed 123000
CUDA_VISIBLE_DEVICES=5 python mobipi/scene_model/collect_images.py --env_name CloseDrawer --layout_id 0 --style_id 0 --seed 123000
```

or download our pre-collect dataset.

### Predictor3: LeLaN (todo)

Todo

### Predictor4: Reachability (todo)

Todo

## Run Benchmark Evaluation

**Basic usage:**

```bash
# name: [CloseDrawer, PnPCounterToCab, CloseDoubleDoor, OpenSingleDoor]
# policy: [bc_transformer, diffusion]
# predictor: [blank, n2m]

# blank (need do again) (PnPCounterToCab 5,6) (CloseDoubleDoor 0,1)
CUDA_VISIBLE_DEVICES=2 python scripts/run_benchmark.py \
  env.name=CloseDrawer \
  env.render=false \
  'env.layout_and_style_ids=[[0,0]]' \
  policy=diffusion \
  predictor=blank \
  benchmark=evaluation \
  benchmark.num_episodes=300

CUDA_VISIBLE_DEVICES=4 python scripts/run_benchmark.py \
  env.name=CloseDrawer \
  env.render=false \
  'env.layout_and_style_ids=[[0,0]]' \
  policy=bc_transformer \
  predictor=blank \
  benchmark=evaluation \
  benchmark.num_episodes=300

# n2m (to do) (PnPCounterToCab 5,6) (CloseDoubleDoor 0,1)
CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmark.py \
  env.name=CloseDrawer \
  env.render=false \
  'env.layout_and_style_ids=[[0,0]]' \
  policy=bc_transformer \
  predictor=n2m \
  predictor.rollout_num=20 \
  benchmark=evaluation \
  benchmark.num_episodes=300

CUDA_VISIBLE_DEVICES=1 python scripts/run_benchmark.py \
  env.name=CloseDrawer \
  env.render=false \
  'env.layout_and_style_ids=[[0,0]]' \
  policy=bc_transformer \
  predictor=n2m \
  predictor.rollout_num=35 \
  benchmark=evaluation \
  benchmark.num_episodes=300

CUDA_VISIBLE_DEVICES=3 python scripts/run_benchmark.py \
  env.name=CloseDrawer \
  env.render=false \
  'env.layout_and_style_ids=[[0,0]]' \
  policy=diffusion \
  predictor=n2m \
  predictor.rollout_num=50 \
  benchmark=evaluation \
  benchmark.num_episodes=300


CUDA_VISIBLE_DEVICES=5 python scripts/run_benchmark.py \
  env.name=CloseDrawer \
  env.render=false \
  'env.layout_and_style_ids=[[0,0]]' \
  policy=diffusion \
  predictor=n2m \
  predictor.rollout_num=20 \
  benchmark=evaluation \
  benchmark.num_episodes=300

CUDA_VISIBLE_DEVICES=6 python scripts/run_benchmark.py \
  env.name=CloseDrawer \
  env.render=false \
  'env.layout_and_style_ids=[[0,0]]' \
  policy=diffusion \
  predictor=n2m \
  predictor.rollout_num=35 \
  benchmark=evaluation \
  benchmark.num_episodes=300

CUDA_VISIBLE_DEVICES=7 python scripts/run_benchmark.py \
  env.name=CloseDrawer \
  env.render=false \
  'env.layout_and_style_ids=[[0,0]]' \
  policy=bc_transformer \
  predictor=n2m \
  predictor.rollout_num=50 \
  benchmark=evaluation \
  benchmark.num_episodes=300





# mobipi (implementing) (CloseDoubleDoor 0,1)
CUDA_VISIBLE_DEVICES=2 python scripts/run_benchmark.py \
  env.name=CloseDrawer \
  env.render=false \
  'env.layout_and_style_ids=[[0,0]]' \
  policy=bc_transformer \
  predictor=mobipi \
  predictor.num_init_samples=2500 \
  predictor.bo_num_samples=500 \
  benchmark=evaluation \
  benchmark.num_episodes=300

CUDA_VISIBLE_DEVICES=4 python scripts/run_benchmark.py \
  env.name=CloseDrawer \
  env.render=false \
  'env.layout_and_style_ids=[[0,0]]' \
  policy=diffusion \
  predictor=mobipi \
  predictor.num_init_samples=2500 \
  predictor.bo_num_samples=505 \
  benchmark=evaluation \
  benchmark.num_episodes=300

# reachability
CUDA_VISIBLE_DEVICES=5 python scripts/run_benchmark.py \
  env.name=CloseDrawer \
  env.render=false \
  'env.layout_and_style_ids=[[0,0]]' \
  policy=diffusion \
  predictor=reachability \
  benchmark=evaluation \
  benchmark.num_episodes=300

CUDA_VISIBLE_DEVICES=6 python scripts/run_benchmark.py \
  env.name=CloseDrawer \
  env.render=false \
  'env.layout_and_style_ids=[[0,0]]' \
  policy=bc_transformer \
  predictor=reachability \
  benchmark=evaluation \
  benchmark.num_episodes=300
```

## Troubleshooting

### OpenGL/GLFW Rendering Issues

If you encounter errors like:

```
libGL error: MESA-LOADER: failed to open swrast: /lib/x86_64-linux-gnu/libLLVM-12.so.1: undefined symbol: ffi_type_sint32
GLFWError: (65543) b'GLX: Failed to create context: BadValue (integer parameter out of range for operation)'
ERROR: could not create window
```

Try following steps

```bash
sudo apt install mlocate
locate libglfw.so.3
# Create soft link libglfw.so.3 you found to /usr/local/lib/
sudo cp /usr/lib/x86_64-linux-gnu/libglfw.so.3 /usr/local/lib/

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
```

## Project Structure

### Folder Structure

```
N2M-benchmark/
├── benchmark/                      # Main benchmark package
│   ├── core/                       # Core benchmark logic
│   │   ├── benchmark_runner.py    # Multi-episode runner
│   │   ├── data_collector.py      # N2M data collection
│   │   └── rollout.py             # Unified rollout loop
│   ├── env/                        # Environment utilities
│   │   └── env_utils.py           # Env creation, config loading
│   ├── policy/                     # Policy wrappers
│   │   ├── base.py                # BasePolicy interface
│   │   ├── robomimic_policy.py    # Robomimic policy wrapper
│   │   ├── vlm_policy.py          # VLM policy wrapper
│   │   └── vla_policy.py          # VLA policy wrapper
│   ├── predictor/                  # Predictor implementations
│   │   ├── base.py                # BasePredictor interface
│   │   ├── blank_predictor.py     # Baseline (no prediction)
│   │   ├── n2m_predictor.py       # N2M predictor
│   │   ├── mobipi_predictor.py    # Mobipi predictor
│   │   ├── lelan_predictor.py     # LeLaN predictor
│   │   └── reachability_predictor.py # Reachability predictor
│   └── utils/                      # Utility modules
│       ├── collision_utils.py     # Collision checking
│       ├── navigation_utils.py    # Teleport navigation
│       ├── obs_utils.py           # Observation extraction
│       ├── observation_utils.py   # RGB-D → point cloud
│       ├── sample_utils.py        # Pose sampling
│       ├── sampling_utils.py      # Target sampling
│       ├── transform_utils.py     # SE2/SE3 transforms
│       └── visualization_utils.py # Video/plot generation
├── configs/                        # Hydra configurations
│   ├── config.yaml                # Main entry point
│   ├── paths/                     # Path configurations
│   │   └── default.yaml           # Data/checkpoint paths
│   ├── env/                       # Environment configs
│   │   ├── configs/               # JSON configs (robosuite)
│   │   │   ├── CloseDrawer.json
│   │   │   ├── CloseDoubleDoor.json
│   │   │   ├── OpenSingleDoor.json
│   │   │   └── PnPCounterToCab.json
│   │   └── robocasa.yaml          # Universal RoboCasa config
│   ├── policy/                    # Policy configs
│   │   ├── configs/               # JSON configs (robomimic)
│   │   │   ├── CloseDrawer_BCtransformer.json
│   │   │   ├── CloseDrawer_diffusion.json
│   │   │   ├── CloseDoubleDoor_BCtransformer.json
│   │   │   ├── CloseDoubleDoor_diffusion.json
│   │   │   ├── OpenSingleDoor_BCtransformer.json
│   │   │   ├── OpenSingleDoor_diffusion.json
│   │   │   ├── pnpCounterToCab_BCtransformer.json
│   │   │   └── PnPCounterToCab_diffusion.json
│   │   ├── bc_transformer.yaml    # BC Transformer config
│   │   └── diffusion.yaml         # Diffusion policy config
│   ├── predictor/                 # Predictor configs
│   │   ├── blank.yaml
│   │   ├── n2m.yaml
│   │   ├── mobipi.yaml
│   │   ├── lelan.yaml
│   │   └── reachability.yaml
│   └── benchmark/                 # Benchmark mode configs
│       ├── collection.yaml        # Data collection mode
│       └── evaluation.yaml        # Evaluation mode
├── data/                           # Data storage (gitignored)
│   ├── policy/                    # Policy checkpoints and datasets
│   │   └── robomimic/
│   │       ├── checkpoints/       # Trained policy checkpoints
│   │       └── datasets/          # Training datasets
│   ├── predictor/                 # Predictor data
│   │   └── n2m/                   # N2M training data
│   │       └── {task}_{layout}_{style}_{policy}/
│   │           ├── pcl/           # Point cloud files
│   │           │   ├── 0.pcd
│   │           │   ├── 1.pcd
│   │           │   └── ...
│   │           └── meta.json      # Metadata (poses, camera params)
│   └── benchmark/                 # Benchmark results
│       └── results/
├── env/                            # Environment submodules
│   ├── robocasa/                  # RoboCasa environment
│   ├── robosuite/                 # Robosuite simulator
│   └── mimicgen/                  # MimicGen
├── policy/                         # Policy submodules
│   └── robomimic/                 # Robomimic policy library
├── predictor/                      # Predictor submodules
│   ├── N2M/                       # N2M predictor
│   ├── mobipi/                    # Mobipi predictor
│   ├── lelan/                     # LeLaN predictor
│   ├── blank/                     # Blank predictor (placeholder)
│   └── reachability/              # Reachability predictor
├── scripts/                        # Executable scripts
│   ├── collect_n2m_data.py        # Collect N2M training data
│   ├── collect_mobipi_images.py   # Collect Mobipi images
│   └── run_benchmark.py           # Run benchmark evaluation
├── docs/                           # Documentation
│   └── RENDERING_ISSUES.md        # Troubleshooting guide
├── install.sh                      # Installation script
├── pyproject.toml                  # Package configuration
└── README.md
```

### System Architecture

```mermaid
graph TB
    A[User] --> B[Hydra Config]
    B --> C[BenchmarkRunner]
    C --> D[Environment]
    C --> E[Predictor]
    C --> F[Policy]
    
    E --> E1[N2M]
    E --> E2[Mobipi]
    E --> E3[LeLaN]
    E --> E4[Blank]
    E --> E5[Reachability]
    
    F --> F1[Robomimic]
    F --> F2[VLM]
    F --> F3[VLA]
    
    D --> D1[RoboCasa]
    
    C --> G[Utils]
    G --> G1[Navigation]
    G --> G2[Collision]
    G --> G3[Sampling]
    G --> G4[Observation]
```



### Workflow

```mermaid
sequenceDiagram
    participant User
    participant Hydra
    participant Runner
    participant Env
    participant Predictor
    participant Policy
    
    User->>Hydra: python scripts/run_benchmark.py
    Hydra->>Runner: Load configs
    Runner->>Env: Create environment
    Runner->>Predictor: Load predictor
    Runner->>Policy: Load policy
    
    loop For each episode
        Runner->>Env: Reset & sample initial pose
        
        loop Until predictor.done
            Runner->>Predictor: predict(obs, pose)
            Predictor-->>Runner: predicted_pose, done
            Runner->>Env: Teleport to pose
        end
        
        loop Until task success/horizon
            Runner->>Policy: predict_action(obs)
            Policy-->>Runner: action
            Runner->>Env: step(action)
        end
        
        Runner->>Runner: Log statistics
    end
    
    Runner->>User: Save results.json
```



## Configuration Architecture

### Configuration Structure

```
configs/
├── env/
│   ├── configs/              # Environment JSON configs (complete robosuite configs)
│   │   ├── CloseDrawer.json
│   │   ├── PnPCounterToCab.json
│   │   └── ...
│   └── robocasa.yaml         # Universal RoboCasa environment config
│
├── policy/
│   ├── configs/              # Policy JSON configs (complete robomimic configs)
│   │   ├── CloseDrawer_BCtransformer.json
│   │   ├── CloseDrawer_diffusion.json
│   │   └── ...
│   ├── bc_transformer.yaml   # BC Transformer policy
│   └── diffusion.yaml        # Diffusion policy
│
├── predictor/                # Predictor configs
│   ├── n2m.yaml
│   ├── mobipi.yaml
│   └── ...
│
└── benchmark/                # Benchmark mode configs
    ├── collection.yaml
    └── evaluation.yaml
```

### Configuration Resolution

```
config.yaml
  └─> env/robocasa.yaml (env.name specified via command line)
       ├─> configs/env/configs/${env.name}.json (full env config)
       └─> policy/bc_transformer.yaml (auto-constructs config_name from env.name)
            └─> configs/policy/configs/${env.name}_BCtransformer.json (full policy config)
```

