#!/bin/bash
# Moment-matching regularization sweep for CD training
# Round 1: Run exps 6, 7, 9 on GPUs 1, 2, 3 simultaneously
# Round 2: Run exps 8, 10 on GPUs 1, 2 after round 1 finishes

echo "=== Round 1: Exps 6, 7, 9 (GPUs 1, 2, 3) ==="

# Exp 6: variance only, weight=0.1 (GPU 1)
python scripts/train_cm.py config/unet_cm_cd_exp6.yaml &
PID6=$!
echo "Exp 6 started (PID $PID6) — var=0.1, GPU 1"

# Exp 7: variance only, weight=1.0 (GPU 2)
python scripts/train_cm.py config/unet_cm_cd_exp7.yaml &
PID7=$!
echo "Exp 7 started (PID $PID7) — var=1.0, GPU 2"

# Exp 9: mu=0.1, var=1.0 (GPU 3)
python scripts/train_cm.py config/unet_cm_cd_exp9.yaml &
PID9=$!
echo "Exp 9 started (PID $PID9) — mu=0.1, var=1.0, GPU 3"

echo ""
echo "Waiting for Round 1 to finish..."
wait $PID6 $PID7 $PID9
echo "=== Round 1 complete ==="

echo ""
echo "=== Round 2: Exps 8, 10 (GPUs 1, 2) ==="

# Exp 8: mu=0.1, var=0.1 (GPU 1)
python scripts/train_cm.py config/unet_cm_cd_exp8.yaml &
PID8=$!
echo "Exp 8 started (PID $PID8) — mu=0.1, var=0.1, GPU 1"

# Exp 10: mu=1.0, var=1.0 (GPU 2)
python scripts/train_cm.py config/unet_cm_cd_exp10.yaml &
PID10=$!
echo "Exp 10 started (PID $PID10) — mu=1.0, var=1.0, GPU 2"

echo ""
echo "Waiting for Round 2 to finish..."
wait $PID8 $PID10
echo "=== Round 2 complete ==="
echo "All moment sweep experiments finished."
