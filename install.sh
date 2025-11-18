#!/bin/bash

set -e  # Exit on error

echo "🔄 Initializing submodules..."
# git submodule update --init --recursive

echo "✅ Installing submodule dependencies..."

# ---- robocasa ----
echo "➡️  Setting up robosuite..."
pip install -e env/robosuite
# pip install robosuite==1.5.0  

echo "➡️  Setting up robocasa..."
pip install -e env/robocasa
(
  cd env/robocasa
  pip install pre-commit
  pre-commit install
  echo "n" | python robocasa/scripts/download_kitchen_assets.py
  echo "n" | python robocasa/scripts/setup_macros.py
)

# ---- predictor lelan ----
# echo "➡️  Installing lelan dependencies..."
# pip install tqdm==4.64.0
# pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
# pip install opencv-python==4.6.0.66
# pip install h5py==3.6.0
# pip install wandb==0.12.18
# pip install prettytable efficientnet-pytorch warmup-scheduler
# pip install diffusers==0.11.1
# pip install openai-clip==1.0.1
# pip install scikit-video==1.1.11
# pip install open3d==0.19.0
# pip install lmdb vit-pytorch positional-encodings
# pip install -e predictor/lelan/train

# ---- policy related ----
echo "🎯 Installing policy-related packages..."
pip install -e env/mimicgen
pip install -e policy/robomimic

# ---- predictor related ----
echo "🎯 Installing predictor-related packages..."
pip install -e predictor/N2M
# pip install -e predictor/mobipi

# ---- benchmark ----
echo "🎯 Installing benchmark package..."
pip install -e .

# ---- Create data directories ----
echo "📁 Creating data directories..."
mkdir -p data/predictor/n2m
mkdir -p data/predictor/mobipi/scene_models
mkdir -p data/predictor/lelan/checkpoints
mkdir -p data/policy/robomimic/{configs,checkpoints,datasets}
mkdir -p data/policy/vlm/checkpoints
mkdir -p data/policy/vla/checkpoints
mkdir -p data/benchmark/results

# ---- Correct a few package versions ----
# Note: nerfstudio and timm have version conflicts, install them separately if needed for mobipi
pip install scikit-optimize cma sentencepiece peft==0.10.0 transformers==4.36.0 huggingface-hub==0.25.0
# Fix numpy version for compatibility
pip install "numpy>=1.23.0,<2.0.0"

# ---- Check PyTorch ----
echo "🔥 Checking PyTorch installation..."
if python -c "import torch" 2>/dev/null; then
  TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
  CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
  echo "✅ PyTorch ${TORCH_VERSION} already installed (CUDA: ${CUDA_AVAILABLE})"
else
  echo "❌ PyTorch not found!"
  echo "Please install PyTorch manually before running this script."
  echo "See A_dos/env_debug.md for installation instructions."
  exit 1
fi
echo "✅ All done!"