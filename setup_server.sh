#!/bin/bash
# ============================================================
# Server Setup Script for GenModeling
# Run this once after cloning the repo on a remote server.
#
# Usage:
#   chmod +x setup_server.sh
#   ./setup_server.sh
# ============================================================

set -e

echo "=== GenModeling Server Setup ==="

# 1. Create conda environment
echo ""
echo "--- Step 1: Creating conda environment ---"
if conda env list | grep -q "gen-modeling"; then
    echo "Environment 'gen-modeling' already exists. Updating..."
    conda env update -f environment.yml --prune
else
    conda env create -f environment.yml
fi

echo ""
echo "--- Step 2: Setting up PYTHONPATH ---"
REPO_DIR=$(pwd)

# Add to bashrc if not already there
if ! grep -q "GenModeling" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# GenModeling project" >> ~/.bashrc
    echo "export PYTHONPATH=\$PYTHONPATH:${REPO_DIR}" >> ~/.bashrc
    echo "Added PYTHONPATH to ~/.bashrc"
else
    echo "PYTHONPATH already in ~/.bashrc"
fi

# Set for current session
export PYTHONPATH=$PYTHONPATH:${REPO_DIR}

echo ""
echo "--- Step 3: Creating data directory ---"
mkdir -p data
mkdir -p darcy_teacher/exp_1/saved_state
mkdir -p darcy_student/exp_1/saved_state

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Activate the environment:  conda activate gen-modeling"
echo ""
echo "  2. Download the data into data/:"
echo "     gdown 'YOUR_GOOGLE_DRIVE_FILE_ID' -O data/2D_DarcyFlow_beta1.0_Train.hdf5"
echo "     (or scp it from your local machine)"
echo ""
echo "  3. Copy your teacher checkpoint:"
echo "     scp local:checkpoint_175.pt darcy_teacher/exp_1/saved_state/"
echo ""
echo "  4. Train teacher (if needed):"
echo "     python scripts/train_vp_diffusion.py config/unet_vp_diffusion.yaml"
echo ""
echo "  5. Train student (CD):"
echo "     python scripts/train_cm.py config/unet_cm_cd.yaml"
echo ""
echo "  6. Generate samples:"
echo "     python scripts/sample_cm.py config/unet_cm_sample.yaml"
echo ""
echo "  TIP: Use tmux so training survives SSH disconnects:"
echo "     tmux new -s train"
echo "     python scripts/train_cm.py config/unet_cm_cd.yaml"
echo "     (Ctrl+B, then D to detach)"
