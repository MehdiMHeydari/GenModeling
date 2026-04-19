# Completed Work

## [TEACHER] Train VP Diffusion teacher (exp_1)
- 400 epochs, cosine schedule, raw-weight training
- Best checkpoint: `darcy_teacher/exp_1/saved_state/checkpoint_200.pt`
- **Outcome:** Teacher used for all downstream distillation. Raw weights beat EMA.

## [CD] Train 4/8/16-step consistency students
- exp_1 = 4-step, exp_2 = 8-step, exp_3 = 16-step
- All initialized from teacher weights
- **Outcome:** 4-step shows mild mode collapse → motivates moment matching. 8 and 16 step are diverse but don't match teacher sharpness.

## [PD] Train Progressive Distillation rounds
- Rounds 1-6: 128 → 64 → 32 → 16 → 8 → 4 → 2 student steps
- **Outcome:** Works as expected, each round halves step count.

## [RF] Train Rectified Flow + Reflow
- RF round 1: trained from random noise-data pairs
- Reflow round 2: trained from deterministically coupled pairs from round 1
- **Outcome:** Reflow straightens trajectories, 1-step sampling viable.

## [MFM] Train Mean Flow Matching
- Dual time embedding `(t, r)`, average velocity over `[t, r]`
- **Outcome:** Works, enables large-step sampling.

## [TEACHER] Teacher sampling diagnostic
- Swept ckpts {200, 399}, steps {75, 250}, samplers {DDIM, Heun}
- **Outcome:**
  - Ckpt 200 >> ckpt 399 on all metrics (confirms prior EMA vs raw finding)
  - Heun does NOT help — DDIM 250 beats Heun 250 on Wasserstein
  - 250 steps visibly better histogram match than 75 steps (peak/tail align)
  - **Decision: don't retrain teacher. Use ckpt 200 + DDIM 250 as canonical.**

## [PAPER] Paper skeleton drafted
- `paper/main.tex` has 5-section structure (Intro / Related / Methods / Experiments / Conclusion)
- Filled with lipsum placeholders for layout review
- **Outcome:** Ready for real text fill-in.

## [EVAL] Lock test indices + noise seeds
- `src/eval/constants.py` defines canonical paper constants (N_TEST_SAMPLES=1000, PAPER_SEEDS=(0,1,2), TEACHER_DDIM_STEPS=250)
- `scripts/lock_test_indices.py` writes `data/test_indices.npy` (1000 held-out samples, deterministic from seed 0)
- **Outcome:** Every paper-facing eval now pulls from the same constants — results across methods and reruns are directly comparable.

## [INFRA] Canonical teacher sampling at 250 DDIM steps
- Bumped teacher DDIM steps from 50/75 to 250 everywhere paper-critical
- Updated `scripts/precompute_teacher_moments.py` to import from constants
- Regenerated `teacher_moments.pt` with 250-step samples
- **Outcome:** Teacher baseline now produces near-perfect histogram match. WD 0.023 at 250 NFE on held-out data.

## [EVAL] Built unified `scripts/evaluate_paper.py`
- One script samples every method kind (teacher, CD, PD, RF, MFM) on locked test indices with locked seeds and writes metrics (MSE, Wasserstein, moment errors, NFE, wall-clock) to CSV.
- Supports `--only` for selective runs and `--seeds` for single-seed quick tests.
- Split eval across 3 GPUs overnight: A=Teacher+CD, B=MM variants, C=RF+Reflow+MFM.
- **Outcome:** `results/eval_all.csv` has 57 rows covering all methods × step counts × 3 seeds. Foundation for every paper table and plot.

## [EVAL] First-round Darcy numeric benchmark
- Ran all methods through the unified eval.
- Headline: Teacher WD 0.023 (250 NFE); RF @ 5 steps WD 0.031 (best few-step); MM-exp22 WD 0.049 @ 16 NFE; CD-16step baseline WD 0.126.
- **Outcome:** Moment matching delivers 2.6× improvement over baseline CD — solid paper result. Surprise finding: direct RF beats distilled methods at lower NFE.

## [BUG] RectifiedFlowSampler step-count bug
- Discovered and fixed: `RectifiedFlowSampler.sample(z, num_steps=N)` takes `num_steps`, but `evaluate_paper.py` and `generate_presentation.py` were passing `t_span_kwargs` (MeanSampler's interface). Silently defaulted to 100 steps.
- All prior RF/Reflow numbers and figures were at 100 steps regardless of requested count.
- Fix: commit 67aed0e. Re-ran RF + Reflow with correct step counts.
- **Outcome:** True step-count sweep for RF now shows quality is best at ~5 steps. Documented in `KNOWLEDGE.md`.
