#!/bin/bash

set -e  # Exit on error

echo "🚀 Setting up Benchmark Environment..."

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script from the benchmark root directory"
    exit 1
fi

# ---- Initialize Git Submodules ----
echo "🔄 Initializing git submodules..."
git submodule update --init --recursive

# ---- Install Dependencies ----
echo "✅ Installing dependencies..."

# Install robosuite first (required by robocasa)
echo "➡️  Installing robosuite..."
pip install robosuite==1.5.0

# Install robocasa
echo "➡️  Installing robocasa..."
pip install -e env/robocasa
(
    cd env/robocasa
    pip install pre-commit
    pre-commit install || true  # Continue if pre-commit fails
    echo "y" | python robocasa/scripts/download_kitchen_assets.py || true
    python robocasa/scripts/setup_macros.py || true
)

# Install mimicgen
echo "➡️  Installing mimicgen..."
pip install -e env/mimicgen

# Install robomimic
echo "➡️  Installing robomimic..."
pip install -e policy/robomimic

# ---- Install PyTorch ----
echo "🔥 Installing PyTorch..."

TORCH_VERSION="2.2.0"
TORCHVISION_VERSION="0.17.0"
TORCHAUDIO_VERSION="2.2.0"

install_torch() {
    if command -v nvidia-smi &> /dev/null; then
        CUDA_MAIN_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1 || echo "0")
        
        echo "Detected CUDA major version: $CUDA_MAIN_VERSION"
        
        if [[ "$CUDA_MAIN_VERSION" -ge 12 ]]; then
            echo "Installing torch with CUDA 12.1"
            pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu121
        elif [[ "$CUDA_MAIN_VERSION" -eq 11 ]]; then
            CUDA_MINOR_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: 11\.\K[0-9]+' | head -1 || echo "0")
            if [[ "$CUDA_MINOR_VERSION" -ge 8 ]]; then
                echo "Installing torch with CUDA 11.8"
                pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu118
            else
                echo "CUDA 11 version too old, falling back to CPU"
                pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION}
            fi
        else
            echo "Unsupported CUDA version, falling back to CPU"
            pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION}
        fi
    else
        echo "CUDA not detected, installing CPU version"
        pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION}
    fi
}

install_torch

# ---- Install Additional Dependencies ----
echo "➡️  Installing additional dependencies..."
pip install tqdm h5py wandb numpy==1.23.3 timm==1.0.12

# Install N2M dependencies (if needed)
if [ -d "baseline/n2m" ]; then
    echo "➡️  Installing N2M dependencies..."
    if [ -f "baseline/n2m/requirements.txt" ]; then
        pip install -r baseline/n2m/requirements.txt || true
    fi
    if [ -f "baseline/n2m/setup.py" ]; then
        pip install -e baseline/n2m || true
    fi
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Configure your environment paths (if needed)"
echo "   2. Download policy checkpoints"
echo "   3. Run evaluations using scripts/eval.py"
echo ""

