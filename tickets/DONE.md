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

## [METRIC] Add structural diversity metric
- Added `structural_diversity` (mean pairwise L2 of per-sample center-of-mass) to `sweep_mm_checkpoints.py` and `evaluate_paper.py`.
- Revealed: CD baseline has catastrophic structural collapse (19% of GT), despite high pixel-level diversity. Prior Wasserstein and pairwise-L2 metrics missed this.
- **Outcome:** MM-exp21 at epoch 75 identified as the real diversity peak (54% of GT). Paper-level reframing: MM is a genuine 2.8× improvement over CD, not a lateral move.

## [BUG] MultistepCDLoss hardcoded 1-channel sample shape
- `sample_moment_loss` created `torch.randn(B, 1, 128, 128)` regardless of model channels
- Broke NS MM training immediately — UNet expected 2 channels
- Fix (commit 0b67c65): added `sample_shape` kwarg to `MultistepCDLoss`, `train_cm.py` passes `tuple(config.unet.dim)`
- **Outcome:** NS MM jobs run; Darcy MM unaffected (default still `(1, 128, 128)`)

## [BUG] Wandb projects hardcoded as `darcy-*` in all train scripts
- All 5 train scripts (`train_vp_diffusion`, `train_cm`, `train_rectified_flow`, `train_mean_flow`, `train_pd`) hardcoded `project="darcy-..."`
- NS training runs launched 2026-04-21/22 ended up in Darcy wandb projects
- Fix (commit 2cc2d47): derive project from `config.dataloader.loader_type`, with `wandb_project` config-level override available
- **Outcome:** NS runs from MM-mm21 onward route to `ns-*` projects. First-wave NS runs (teacher/RF/MFM) left in Darcy projects to avoid losing training continuity — documented in `KNOWLEDGE.md`

## [INFRA] NS dataset pipeline
- Downloaded 2 PDEBench files (`ns_incom_inhom_2d_512-0/1.h5`, ~9 GB each)
- `scripts/preprocess_ns.py` flattens time dim, downsamples 512→128 via 4×4 avg pool, subsamples to fixed count (seed-locked)
- Final dataset: ~8000 frames at (2, 128, 128), saved as `data/ns_incom_128_merged.h5`
- `config/ns_teacher.yaml` set to 7000 train / 1000 test split (matches Darcy test count)
- **Outcome:** NS data ready for the same pipeline as Darcy

## [INFRA] NS teacher + downstream training kicked off
- Teacher (600 ep), RF round 1 (800 ep), Reflow (400 ep), MFM (1000 ep), CD-16 (1000 ep) chained where appropriate
- MM-mm21 (mu=4, var=200) and MM-mm22 (mu=16, var=150) launched against teacher checkpoint_75 + precomputed moments
- All 5 NS jobs running concurrently across GPUs 0/1/3/4/5
- **Outcome:** Full NS sweep underway

## [INFRA] NS Teacher fully trained (2026-04-22)
- 600 epochs in ~14 hours on GPU 3 (A100)
- Final checkpoint: `ns_teacher/exp_1/saved_state/checkpoint_599.pt`
- Teacher moments precomputed from checkpoint_75 (early — should consider recomputing from final ckpt if MM results look bad)
- **Outcome:** Teacher available as DDIM baseline for NS eval; CD-16 chain auto-started after

## [INFRA] NS RF round 1 fully trained (2026-04-22)
- 800 epochs in ~21 hours on GPU 0 (A100)
- Final checkpoint: `ns_rectified_flow/exp_1/saved_state/checkpoint_799.pt`
- **Outcome:** RF baseline available; reflow blocked on bug fix (now resolved)

## [BUG] generate_reflow_pairs hardcoded 1-channel UNet
- Same bug class as MultistepCDLoss — built UNet with 1-channel default
- Crashed NS reflow pair generation immediately after RF round 1 finished
- Fix (commit e7eaa3f): added `--channels` CLI flag (default 1, NS uses 2);
  updated `run_ns_rf_pipeline.sh` to pass `--channels 2`
- **Outcome:** NS reflow can now run after restarting from RF round 1's checkpoint

## [BUG] RectifiedFlowSampler step-count bug
- Discovered and fixed: `RectifiedFlowSampler.sample(z, num_steps=N)` takes `num_steps`, but `evaluate_paper.py` and `generate_presentation.py` were passing `t_span_kwargs` (MeanSampler's interface). Silently defaulted to 100 steps.
- All prior RF/Reflow numbers and figures were at 100 steps regardless of requested count.
- Fix: commit 67aed0e. Re-ran RF + Reflow with correct step counts.
- **Outcome:** True step-count sweep for RF now shows quality is best at ~5 steps. Documented in `KNOWLEDGE.md`.

## [TEACHER] NS teacher sampling diagnostic (2026-04-24)
- Three-phase sweep in `diagnostics/ns_teacher_v{1,2,3}/`:
  - v1: ckpts {200, 400, 599} × steps {75, 250} × samplers {DDIM, Heun}
  - v2: intermediate ckpts {50, 75, 100, 150, 200} × DDIM 75
  - v3: full spread {50, 75, 100, 150, 200, 599} × DDIM 75, with random GT sampling + per-channel (Vx, Vy) histograms
- Findings:
  - **NS training is wildly non-monotone**: WD curve is 50=0.011 → 75=0.014 → 100=0.111 → 150=0.051 → 200=0.033 → 400=0.087 → 599=0.041. Looks like an LR/EMA interaction; good enough for paper but can't trust latest checkpoints without checking.
  - **More DDIM steps doesn't help** on NS (75 ≈ 250 within 3%) — opposite of Darcy where 250 >> 75.
  - **Heun doesn't help** (matches Darcy finding).
  - |v| magnitude histograms overstate error because magnitude of a noisy 2D vector is biased upward near 0; per-channel (Vx, Vy) histograms look much tighter.
  - **Canonical NS teacher: `checkpoint_75.pt` + DDIM 75**. Locked in `src/eval/constants.py` NS block.
- **Outcome:** MM moments already precomputed from ckpt 75 → MM-mm21/22 do not need to restart.

## [EVAL] NS eval pipeline (2026-04-24)
- `src/eval/constants.py` parameterized by dataset via `_DATASETS` dict + `get_dataset(name)` helper; existing Darcy module constants preserved for back-compat.
- `scripts/lock_ns_test_indices.py` writes 1000 locked NS test indices.
- `scripts/evaluate_paper.py` now takes `--dataset ns|darcy` (or reads `dataset:` at the top of the config); UNet config built per-call from `data_shape`; distribution-shape metrics (Wasserstein, skew, kurt) and `structural_diversity` all reduce through `to_scalar_field` which is first-channel for Darcy and `sqrt(Vx² + Vy²)` for NS.
- CSV gained a `dataset` column — back-compat guard errors out if the output path points at an old-header CSV rather than silently appending mismatched rows.
- `config/ns_paper_eval.yaml` enumerates 14 evaluations covering teacher + RF ×3 ckpts + Reflow ×2 + MFM ×3 + CD-16 ×3 + MM variants. Spreads early/mid/late per method because training is non-monotone.
- **Outcome:** Unified pipeline — same script runs Darcy and NS. Ready to run NS eval once `ns_test_indices.npy` is generated on the GPU server.

## [BUG] Wasserstein cap-bias on flattened-array (2026-04-26)
- `evaluate_paper.py` and `diagnose_ns_teacher.py` both capped flattened (N, H, W) arrays at 100k/50k by taking the **first N** elements. Because the array is sample-major, the first N elements live in the first few frames. With NS test indices being random-but-sorted within the 1000-frame tail, those first frames were temporally clustered → narrow |v| distribution → WD inflated by 10-30× across all methods.
- Symptom: every NS method got WD ≈ 0.45-0.55, indistinguishable. Teacher gave WD 0.49 in eval but 0.014 in the diagnostic on the same model.
- Fix (commit eaa24c5): random subsample with `np.random.RandomState(0)` — reproducible and unbiased.
- **Outcome:** NS eval becomes interpretable. Teacher WD drops from 0.49 to 0.028 (matches diagnostic ballpark), all other methods become discriminable.

## [EVAL] First-round NS numeric benchmark (2026-04-27)
- Ran 18 (method × ckpt) entries through the unified eval after the cap-bias fix. See `tickets/KNOWLEDGE.md` for the full table.
- **Headline:** Reflow @ 1 NFE wins (WD 0.0102, 113% of teacher diversity), beating teacher @ 75 NFE (WD 0.028) by 2.7×. Reflow is fully converged at 1 step.
- **MM lifts CD diversity** 88% → 95% of teacher (smaller than Darcy's 19% → 54%, because CD on NS isn't catastrophic to begin with).
- **MM costs WD** (0.026 → 0.081) — diversity intervention, not quality intervention.
- **The Darcy "RF > Reflow" finding does NOT generalize** to NS — exact opposite holds.
- **Outcome:** Two-dataset benchmark complete. Paper can claim Reflow as the strong NS baseline and MM as a CD-collapse fix. Now have legitimate cross-dataset comparison.
