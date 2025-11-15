# Benchmark Environment for Mobile Manipulation

A unified benchmark environment for evaluating different navigation-to-manipulation methods, including N2M, mobipi, and other baselines.

## Overview

This benchmark provides a standardized framework to compare different approaches for mobile manipulation tasks, specifically focusing on the problem of finding optimal robot base poses for manipulation policies.

## Project Structure

```
.
├── env/                    # Environment modules (from original repositories)
│   ├── robocasa/          # RoboCasa simulation environment
│   ├── robosuite/         # RoboSuite base library
│   └── mimicgen/          # MimicGen data generation
│
├── policy/                 # Policy implementations
│   ├── robomimic/         # RoboMimic policy suite
│   ├── VLA/               # Vision-Language-Action policies (placeholder)
│   └── VLM/               # Vision-Language Model policies (placeholder)
│
├── baseline/               # Baseline methods (unified interface)
│   ├── n2m/               # N2M navigation method (nav2man module)
│   ├── mobipi/            # Mobi-π method
│   └── other/             # Other baseline methods
│
└── scripts/               # Benchmark scripts and utilities
    └── eval.py            # Main evaluation script
```

## Installation

### Prerequisites

- Python 3.10
- CUDA (for GPU support)
- Git

### Setup

1. Create a conda environment:
```bash
mamba create -c conda-forge -n benchmark python=3.10
mamba activate benchmark
```

2. Clone this repository with submodules:
```bash
git clone --recurse-submodules <repository-url>
cd benchmark
```

Or if you've already cloned without submodules:
```bash
git submodule update --init --recursive
```

3. Run the installation script:
```bash
chmod +x install.sh
./install.sh
```

The installation script will:
- Initialize git submodules (robocasa, robosuite, mimicgen, robomimic, N2M, mobipi)
- Install dependencies
- Set up the benchmark environment
