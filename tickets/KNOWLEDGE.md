# Knowledge Base

Non-obvious findings we've learned. Check here before changing a default or re-investigating.

## Teacher sampling

- **Canonical teacher config**: `checkpoint_200.pt` + **DDIM 250 steps**. Do not change without re-running the diagnostic sweep.
- **Raw weights beat EMA** (0.9999 decay) across all epochs for this teacher. All downstream uses `model_state_dict`, not `ema_state_dict`.
- **Later checkpoints are worse**: ckpt 399 has 25% higher Wasserstein than ckpt 200 → raw weights drift/overfit past epoch 200.
- **Heun sampler does NOT help**. Tested: Heun 250 steps (499 NFE) has worse Wasserstein than DDIM 250 steps. Don't waste NFE on higher-order samplers for this model.
- **250 steps is the sweet spot**. 75 steps visibly undershoots the mode and has too-heavy right tail. 250 matches the histogram near-perfectly.

## Training dependencies

- **Teacher-independent** (train from ground truth, can run in parallel with teacher retraining):
  - Rectified Flow, Reflow, Mean Flow Matching
- **Teacher-dependent** (blocked on teacher finishing):
  - Progressive Distillation, Consistency Distillation, Moment Matching

## CD training internals

- CD training uses its own `N_teacher` annealing schedule (64 → 1280 over 100k iters). Teacher is called single-pass (`predict_x`) per training step, **not** multi-step DDIM sampling. So CD training is unaffected by `DDIM_STEPS` constants in sampling scripts.

## Known failure modes

- **CD 4-step mode-collapses** — main motivation for the moment-matching objective.
- **Teacher moments were computed at 50 DDIM steps** (stale) — any moment-matching run before regenerating `teacher_moments.pt` is on incorrect targets.

## Dataset

- 2D Darcy Flow: 9000 train, ~1000 test (exact test set needs locking for paper).
- Normalization: min-max to `[-1, 1]`, stats saved in `darcy_teacher/exp_1/saved_state/data_{min,max}.npy`.

## Evaluation conventions (for paper)

- **Test sample count**: 1000 (all available held-out samples; matches DiffusionPDE's 1000-scene convention).
- **Seeds per config**: 3 (diffusion-paper standard; produces error bars).
- **Locked test indices**: store in `data/test_indices.npy`, use across every eval run.
- **Metric to trust for sample quality**: 1-Wasserstein on marginal pixel distribution.
- **Metric to NOT trust naively**: pixel MSE — compares unpaired generated sample to unpaired GT sample, so reflects magnitude more than quality.
