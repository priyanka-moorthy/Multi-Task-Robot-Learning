#!/bin/bash
# Installation script for Multi-Task Robot Learning
# Handles Apple Silicon (ARM64) compatibility

set -e  # Exit on error

echo "=================================="
echo "Multi-Task Robot Learning Setup"
echo "=================================="

# Detect architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Install dependencies in correct order
echo ""
echo "[1/4] Installing base dependencies..."
python3 -m pip install numpy pillow pyyaml tqdm matplotlib

echo ""
echo "[2/4] Installing deep learning libraries..."
python3 -m pip install torch torchvision transformers timm

echo ""
echo "[3/4] Installing RL environment..."
python3 -m pip install gymnasium

echo ""
echo "[4/4] Installing physics simulator..."
if [ "$ARCH" = "arm64" ]; then
    echo "Apple Silicon detected - installing MuJoCo (recommended)"
    python3 -m pip install mujoco dm-control
    echo "✓ MuJoCo installed successfully"
else
    echo "x86_64 detected - you can use PyBullet or MuJoCo"
    read -p "Install PyBullet (p) or MuJoCo (m)? [m]: " choice
    choice=${choice:-m}

    if [ "$choice" = "p" ]; then
        python3 -m pip install pybullet
        echo "✓ PyBullet installed"
    else
        python3 -m pip install mujoco dm-control
        echo "✓ MuJoCo installed"
    fi
fi

echo ""
echo "=================================="
echo "✓ Installation complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Test environment: python test_environment.py"
echo "  2. Continue tutorial: Ready for Step 5 (Vision Encoder)"
