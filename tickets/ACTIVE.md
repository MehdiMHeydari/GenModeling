# Active Work

> ⚠️ **wandb mixing note**: NS teacher, RF, MFM runs launched 2026-04-21/22
> log to the `darcy-*` wandb projects (see KNOWLEDGE.md). MM-mm21,
> MM-mm22, and any NS run launched after 2026-04-22 ~13:00 route to
> `ns-*` correctly.

## In flight — NS training

### [INFRA] GPU 3: CD-16 chain (epoch ~350/1000)
- Tmux: `ns_teacher_cd`
- Teacher 600 ep finished (`ns_teacher/exp_1/saved_state/checkpoint_599.pt`)
- CD-16 student now training, on track at ~2.1 min/epoch
- Output: `ns_student/exp_1/saved_state/`

### [INFRA] GPU 4: NS MFM (epoch ~175/1000)
- Tmux: `ns_mfm`
- Slow due to batch 8 + JVP (we dropped batch from 16 → 8 due to OOM)
- Output: `ns_mean_flow/exp_1/saved_state/`

### [INFRA] GPU 5: NS MM-mm21 (mu=4, var=200) — VERY SLOW
- Tmux: `ns_moments`
- Epoch 25/300 after ~12 hours (~28 min/epoch — 12× slower than vanilla CD)
- Slow because `moment_every: 1` runs full sampling chain every iteration
- Plan: let it cook overnight, evaluate checkpoint_75 (Darcy's diversity peak) tomorrow

### [INFRA] GPU 1: NS MM-mm22 (mu=16, var=150) — slow same reason
- Tmux: `ns_mm22`
- Epoch 25/300 similar pace
- Same plan as mm21

## Broken — needs restart

### [INFRA] NS Reflow round 2
- `generate_reflow_pairs.py` was 1-channel hardcoded → crashed
- Fixed in commit e7eaa3f (added `--channels` flag)
- RF round 1 (epoch 799) is done, no need to retrain
- Needs to: regenerate pairs with `--channels 2`, then train Reflow round 2
- Suggested: GPU 0 (now free) — see "next steps" below

## Queued

### [INFRA] Restart NS Reflow
```
cd ~/GenModeling-main && git pull
tmux new -s ns_reflow
conda activate gen-modeling
CUDA_VISIBLE_DEVICES=0 ./scripts/run_ns_rf_pipeline.sh
```
Note: the pipeline script will see RF round 1 already done (checkpoint_799 exists) and... actually, it WILL retrain unless we modify it. Either modify the script to skip step 1, or run pair gen + round 2 manually.

### [EVAL] Extend pipeline to 2-channel data
- `src/eval/constants.py` has `DATA_SHAPE = (1, 128, 128)` hardcoded — parametrize
- `evaluate_paper.py`, `generate_paper_figures.py` also assume 1 channel
- `structural_diversity` metric: decide between per-channel-then-average OR magnitude field `sqrt(Vx² + Vy²)`
- Required before running NS eval

### [EVAL] Lock NS test indices
- `data/ns_test_indices.npy` with 1000 indices from the 8000-frame pool
- Variant of `scripts/lock_test_indices.py` for NS

### [EVAL] Run NS numeric eval
- Create `config/ns_paper_eval.yaml` listing every NS method
- Run `scripts/evaluate_paper.py --config config/ns_paper_eval.yaml`
- Output: `results/ns_eval_all.csv`

### [PAPER] Generate NS sample grid + histograms
- Extend `scripts/generate_paper_figures.py` for 2-channel NS

### [PAPER] Fill in `paper/main.tex` Methods section
- Can be drafted now; doesn't depend on numbers

### [DECISION] Restart NS MM on better teacher checkpoint?
- Currently against teacher checkpoint_75
- Now we have checkpoint_599 (final) — could re-precompute moments and restart MM
- Decide based on what current MM checkpoints look like at evaluation
