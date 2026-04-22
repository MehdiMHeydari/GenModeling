#!/bin/bash
# =============================================================================
# NS Rectified Flow Pipeline (round 1 + reflow round 2)
#
# Round 1 trains for 800 epochs. Once checkpoint at epoch 400 is ready,
# generates reflow pairs and starts round 2 on the same GPU after round 1
# finishes (or in parallel if you split GPUs manually).
#
# Simpler than the Darcy version — runs everything sequentially on one GPU
# so you don't need to coordinate multiple processes.
#
# Usage:
#   chmod +x scripts/run_ns_rf_pipeline.sh
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_ns_rf_pipeline.sh
# =============================================================================

set -e

ROUND1_CONFIG="config/ns_rectified_flow.yaml"
REFLOW_CONFIG="config/ns_rectified_flow_reflow.yaml"

ROUND1_FINAL_CKPT="ns_rectified_flow/exp_1/saved_state/checkpoint_799.pt"
REFLOW_PAIRS_PATH="ns_rectified_flow/reflow_pairs.pt"

echo "============================================"
echo " NS Rectified Flow Pipeline"
echo "============================================"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-default}"
echo ""

# --- Step 1: Round 1 training ---
echo "[1/3] Round 1 (800 epochs)..."
python scripts/train_rectified_flow.py "$ROUND1_CONFIG"

if [ ! -f "$ROUND1_FINAL_CKPT" ]; then
    echo "ERROR: expected $ROUND1_FINAL_CKPT after round 1. Aborting."
    exit 1
fi

# --- Step 2: Generate reflow pairs ---
echo ""
echo "[2/3] Generating reflow pairs from $ROUND1_FINAL_CKPT..."
python scripts/generate_reflow_pairs.py \
    --checkpoint "$ROUND1_FINAL_CKPT" \
    --n_pairs 7000 \
    --ode_steps 100 \
    --save_path "$REFLOW_PAIRS_PATH" \
    --gpu 0

# --- Step 3: Reflow (round 2) training ---
echo ""
echo "[3/3] Reflow training (400 epochs)..."
python scripts/train_rectified_flow.py "$REFLOW_CONFIG"

echo ""
echo "============================================"
echo " NS RF Pipeline complete"
echo "============================================"
