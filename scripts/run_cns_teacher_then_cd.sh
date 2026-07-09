#!/bin/bash
# =============================================================================
# CNS Teacher → CD Chain
#
# Runs the CNS VP-diffusion teacher to completion (600 epochs), then
# automatically starts CNS consistency distillation (16-step student) on the
# same GPU. Total runtime ~2 days sequential on A100.
#
# Usage:
#   chmod +x scripts/run_cns_teacher_then_cd.sh
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_cns_teacher_then_cd.sh
# =============================================================================

set -e

TEACHER_CONFIG="config/cns_teacher.yaml"
CD_CONFIG="config/cns_cm_cd.yaml"
TEACHER_FINAL_CKPT="cns_teacher/exp_1/saved_state/checkpoint_599.pt"

echo "============================================"
echo " CNS Teacher → CD Chain"
echo "============================================"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-default}"
echo ""

echo "[1/2] Training CNS teacher (600 epochs)..."
python scripts/train_vp_diffusion.py "$TEACHER_CONFIG"

if [ ! -f "$TEACHER_FINAL_CKPT" ]; then
    echo "ERROR: expected $TEACHER_FINAL_CKPT to exist after teacher training."
    echo "Teacher training may have failed. Aborting CD."
    exit 1
fi

echo ""
echo "Teacher training complete. Final checkpoint: $TEACHER_FINAL_CKPT"
echo ""

echo "[2/2] Training CNS CD-16step student (1000 epochs)..."
python scripts/train_cm.py "$CD_CONFIG"

echo ""
echo "============================================"
echo " CNS Teacher + CD pipeline complete"
echo "============================================"
