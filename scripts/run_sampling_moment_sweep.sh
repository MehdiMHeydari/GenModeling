#!/bin/bash
# Sampling-based moment matching sweep
# Run teacher moment pre-computation first, then launch 3 experiments

echo "=== Step 1: Pre-compute teacher moments ==="
python scripts/precompute_teacher_moments.py --gpu 0 --n_samples 1000

echo ""
echo "=== Step 2: Launch experiments on 3 GPUs ==="
echo "  exp 11: GPU 1, var_weight=5.0  (~21% of CD)"
echo "  exp 12: GPU 2, var_weight=10.0 (~42% of CD)"
echo "  exp 13: GPU 3, mu_weight=1.0 + var_weight=25.0 (~225% of CD)"
echo ""

# Launch all 3 in background
python scripts/train_cm.py config/unet_cm_cd_exp11.yaml &
python scripts/train_cm.py config/unet_cm_cd_exp12.yaml &
python scripts/train_cm.py config/unet_cm_cd_exp13.yaml &

echo "All 3 experiments launched. Monitor on W&B."
echo "PIDs: $(jobs -p)"
wait
echo "All experiments complete."
