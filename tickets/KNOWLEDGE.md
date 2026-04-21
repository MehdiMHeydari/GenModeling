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

## Sampler interface gotcha (commit 67aed0e)

Each sampler in `src/inference/samplers.py` has a different step-count kwarg — passing the wrong name silently uses a default, not an error:

| Sampler | Correct kwarg |
|---------|---------------|
| `RectifiedFlowSampler.sample(z, num_steps=N)` | `num_steps` (default 100) |
| `MeanSampler.sample(z, t_span_kwargs={"start":0,"end":1,"steps":N+1})` | `t_span_kwargs` |
| `MultistepCMSampler.sample(z)` | none — step count is baked into `model.student_steps` |

**Why this is dangerous:** all samplers take `**kwargs`, so passing e.g. `t_span_kwargs` to `RectifiedFlowSampler` compiles fine but silently falls back to `num_steps=100`. All RF/Reflow runs before commit 67aed0e were secretly at 100 steps regardless of the requested count. Presentation figures and the first `evaluate_paper.py` run are both affected.

## Dataset

- 2D Darcy Flow: 9000 train, ~1000 test (exact test set needs locking for paper).
- Normalization: min-max to `[-1, 1]`, stats saved in `darcy_teacher/exp_1/saved_state/data_{min,max}.npy`.

## Evaluation conventions (for paper)

- **Test sample count**: 1000 (all available held-out samples; matches DiffusionPDE's 1000-scene convention).
- **Seeds per config**: 3 (diffusion-paper standard; produces error bars).
- **Locked test indices**: store in `data/test_indices.npy`, use across every eval run.
- **Metric to trust for sample quality**: 1-Wasserstein on marginal pixel distribution.
- **Metric to NOT trust naively**: pixel MSE — compares unpaired generated sample to unpaired GT sample, so reflects magnitude more than quality.

## Diversity metrics — 1D Wasserstein is not enough (2026-04-21)

Mode collapse in distilled few-step generators is not caught by 1D Wasserstein
on flattened pixels, nor by pairwise-L2 pixel diversity. Both can be fooled
by "same-shape-different-size" mode collapse.

**The right metric**: `structural_diversity` = mean pairwise L2 of per-sample
center-of-mass. High `pix` + low `struct` = structurally collapsed.

Example, epoch 999 checkpoints on Darcy:
- GT: pix=15.80, struct=12.40
- Teacher: pix=16.14, struct=9.56 (77% of GT) — diverse
- CD baseline: pix=18.38, struct=2.33 (**19% of GT**) — catastrophically collapsed despite high pix
- MM-exp21 (mu=4, var=200) @ epoch 75: pix=10.23, struct=6.65 (54% of GT) — best student

**MM is a legitimate fix** for CD's structural collapse (2.8× improvement),
but **only at the diversity peak around epoch 75**. Later checkpoints
collapse again because the 1D moment loss aggregates away spatial
information and can be satisfied by narrow output modes.

## MM loss is spatially blind

The moment loss computes four scalars:
- mean/var of per-sample spatial mean
- mean/var of per-sample spatial variance

All spatial structure is aggregated away before the loss is evaluated.
This means two totally different-looking batches (e.g. varied-shape vs
centered-blobs-with-varied-intensity) can have identical moments. See
commit logs and `scripts/sweep_mm_checkpoints.py` discussion for details.
Future work: spatially-aware moment loss (per-pixel moments, spatial power
spectrum, or explicit COM penalty) — tracked in BACKLOG.md.

## First-round benchmark findings (Darcy, 2026-04-19)

Ranked by Wasserstein (lower = better); all at 1000 test samples × 3 seeds.

| Method | Best step count | NFE | Wasserstein (mean) |
|--------|-----------------|-----|--------------------|
| Teacher | 250 | 250 | 0.023 |
| RF | 5 | 5 | **0.031** (best few-step) |
| RF | 1 | 1 | 0.039 |
| MM-exp22 (mu=16, var=150) | 16 | 16 | 0.049 |
| MM-exp21 (mu=4, var=200) | 16 | 16 | 0.050 |
| Reflow | any (1-10) | 1-10 | ~0.068 |
| MFM | any (2-16) | 2-16 | ~0.075 |
| CD-16step (baseline) | 16 | 16 | 0.126 |
| CD-8step | 8 | 8 | 0.231 |
| CD-4step | 4 | 4 | 0.282 |

**Paper-level implications:**
- **Moment matching works**: 2.6× WD improvement over baseline CD-16step — the main contribution claim holds.
- **RF is strongest few-step method** on Darcy — MM story is not "we beat everything" but "we fix CD". Direct-training flows may deserve more attention than distillation for PDE surrogates.
- **Reflow and MFM have flat step-count response** — both unexpected. Listed as open investigations in BACKLOG.md.
- **Never claim "X beats Y" from one dataset** — user requires at least one more PDE dataset (Navier-Stokes / Burgers) before paper conclusions.
