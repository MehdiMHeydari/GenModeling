#!/bin/bash
# =============================================================================
# Rectified Flow Pipeline: Round 1 + Reflow (Round 2) in one go
#
# Round 1 trains to 800 epochs on the specified GPU.
# Once checkpoint at epoch 400 is ready, it auto-detects the least-used
# GPU for reflow pair generation + round 2 training.
#
# Usage:
#   chmod +x scripts/run_rectified_flow_pipeline.sh
#   ./scripts/run_rectified_flow_pipeline.sh        # auto-detect GPU for round 1 too
#   ./scripts/run_rectified_flow_pipeline.sh 3       # force round 1 on GPU 3
# =============================================================================

set -e

ROUND1_CONFIG="config/unet_rectified_flow.yaml"
REFLOW_CONFIG="config/unet_rectified_flow_reflow.yaml"

ROUND1_CKPT_DIR="darcy_rectified_flow/exp_1/saved_state"
REFLOW_TRIGGER_CKPT="${ROUND1_CKPT_DIR}/checkpoint_400.pt"
REFLOW_PAIRS_PATH="darcy_rectified_flow/reflow_pairs.pt"

# --- GPU selection helper ---
find_free_gpu() {
    # Returns the GPU index with the most free memory.
    # Parses nvidia-smi to find GPU with lowest memory usage.
    local exclude_gpu="${1:--1}"
    local best_gpu=""
    local best_free=0

    while IFS=, read -r idx used total; do
        # Strip whitespace
        idx=$(echo "$idx" | tr -d ' ')
        used=$(echo "$used" | tr -d ' MiB')
        total=$(echo "$total" | tr -d ' MiB')
        free=$((total - used))

        if [ "$idx" != "$exclude_gpu" ] && [ "$free" -gt "$best_free" ]; then
            best_free=$free
            best_gpu=$idx
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)

    # If no other GPU found (single GPU system), fall back to any GPU with most free mem
    if [ -z "$best_gpu" ]; then
        best_gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | sort -t, -k2 -rn | head -1 | cut -d, -f1 | tr -d ' ')
    fi

    echo "$best_gpu"
}

echo "============================================"
echo " Rectified Flow Pipeline"
echo "============================================"
echo ""

# --- Pick GPU for round 1 ---
if [ -n "$1" ]; then
    GPU_ROUND1=$1
else
    GPU_ROUND1=$(find_free_gpu)
fi
echo "Round 1 GPU: $GPU_ROUND1"

# --- Step 1: Start Round 1 training in background ---
echo ""
echo "[1/4] Starting Round 1 training (800 epochs, background on GPU $GPU_ROUND1)..."
CUDA_VISIBLE_DEVICES=$GPU_ROUND1 python scripts/train_rectified_flow.py "$ROUND1_CONFIG" &
ROUND1_PID=$!
echo "  Round 1 PID: $ROUND1_PID"

# --- Step 2: Wait for round 1 to reach epoch 400 ---
echo ""
echo "[2/4] Waiting for round 1 to reach epoch 400..."
echo "  Watching for: $REFLOW_TRIGGER_CKPT"

while [ ! -f "$REFLOW_TRIGGER_CKPT" ]; do
    # Check if round 1 is still running
    if ! kill -0 "$ROUND1_PID" 2>/dev/null; then
        echo "  ERROR: Round 1 training exited before reaching epoch 400."
        echo "  Check logs for errors."
        exit 1
    fi
    sleep 60
done

echo "  Checkpoint found! Round 1 reached epoch 400."
echo "  (Round 1 continues training to 800 in background on GPU $GPU_ROUND1)"

# --- Pick a different GPU for reflow ---
GPU_REFLOW=$(find_free_gpu "$GPU_ROUND1")
echo "  Reflow GPU: $GPU_REFLOW"

# --- Step 3: Generate reflow pairs ---
echo ""
echo "[3/4] Generating reflow pairs from epoch 400 checkpoint (GPU $GPU_REFLOW)..."
CUDA_VISIBLE_DEVICES=$GPU_REFLOW python scripts/generate_reflow_pairs.py \
    --checkpoint "$REFLOW_TRIGGER_CKPT" \
    --n_pairs 9000 \
    --ode_steps 100 \
    --save_path "$REFLOW_PAIRS_PATH" \
    --gpu 0

echo "  Reflow pairs saved to $REFLOW_PAIRS_PATH"

# --- Step 4: Start Round 2 (reflow) training ---
echo ""
echo "[4/4] Starting Round 2 (reflow) training (400 epochs, GPU $GPU_REFLOW)..."
CUDA_VISIBLE_DEVICES=$GPU_REFLOW python scripts/train_rectified_flow.py "$REFLOW_CONFIG"

echo ""
echo "============================================"
echo " Reflow training complete!"
echo "============================================"
echo ""

# Wait for round 1 to finish if still running
if kill -0 "$ROUND1_PID" 2>/dev/null; then
    echo "Round 1 is still training in background (PID: $ROUND1_PID)."
    echo "Waiting for it to finish..."
    wait "$ROUND1_PID"
    echo "Round 1 complete."
else
    echo "Round 1 already finished."
fi

echo ""
echo "All done. Checkpoints:"
echo "  Round 1: $ROUND1_CKPT_DIR/"
echo "  Reflow:  darcy_rectified_flow_reflow/exp_1/saved_state/"
