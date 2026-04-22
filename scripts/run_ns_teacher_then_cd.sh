#!/bin/bash
# =============================================================================
# NS Teacher → CD Chain
#
# Runs the NS VP-diffusion teacher to completion (400 epochs), then
# automatically starts NS consistency distillation (16-step student) on the
# same GPU. Total runtime ~2 days sequential.
#
# Usage:
#   chmod +x scripts/run_ns_teacher_then_cd.sh
#   CUDA_VISIBLE_DEVICES=3 ./scripts/run_ns_teacher_then_cd.sh
# =============================================================================

set -e

TEACHER_CONFIG="config/ns_teacher.yaml"
CD_CONFIG="config/ns_cm_cd.yaml"
TEACHER_FINAL_CKPT="ns_teacher/exp_1/saved_state/checkpoint_399.pt"

echo "============================================"
echo " NS Teacher → CD Chain"
echo "============================================"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-default}"
echo ""

# --- Step 1: Train teacher ---
echo "[1/2] Training NS teacher (400 epochs)..."
python scripts/train_vp_diffusion.py "$TEACHER_CONFIG"

# --- Step 2: Verify teacher finished ---
if [ ! -f "$TEACHER_FINAL_CKPT" ]; then
    echo "ERROR: expected $TEACHER_FINAL_CKPT to exist after teacher training."
    echo "Teacher training may have failed. Aborting CD."
    exit 1
fi

echo ""
echo "Teacher training complete. Final checkpoint: $TEACHER_FINAL_CKPT"
echo ""

# --- Step 3: Train CD ---
echo "[2/2] Training NS CD-16step student (1000 epochs)..."
python scripts/train_cm.py "$CD_CONFIG"

echo ""
echo "============================================"
echo " NS Teacher + CD pipeline complete"
echo "============================================"
