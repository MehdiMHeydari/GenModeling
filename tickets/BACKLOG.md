# Backlog

## [EVAL] Build unified `scripts/evaluate_all.py`
- One script runs every method (Teacher, PD, CD, RF/Reflow, MFM, MomentMatched) on the same test batch
- Fixed test indices from `data/test_indices.npy` (to be created)
- Fixed noise seeds across methods for apples-to-apples
- **3 seeds per config**, report mean ± std (matches DiffusionPDE convention)
- 500 test samples (aligns with DiffusionPDE's 1000, PhysicsNeMo's 100)
- Emits `results/eval_all.csv` with columns:
  `method, config, step_count, seed, pixel_mse, wasserstein, mean_err, std_err, skew_err, kurt_err, mmd, nfe, wall_clock_s`

## [PAPER] Paper figures generation script
- `scripts/make_paper_figures.py` that reads `results/eval_all.csv` and produces:
  - Table 1: all methods × metrics at matched NFE
  - Fig 1: Pareto (Wasserstein vs NFE, one curve per method)
  - Fig 2: Moment errors across step counts
  - Fig 3: Moment matching ablation (CD vs CD+MM)
  - Fig 4: Qualitative grid
  - Fig 5: Marginal histograms

## [PAPER] Fill in paper body
- `paper/main.tex` currently has lipsum
- Order: Methods → Experiments → Related Work → Intro/Abstract/Conclusion last

## [INFRA] Second PDE dataset: Navier-Stokes (required before paper conclusions)
- User explicitly wants a second dataset before claiming "X beats Y"
- **Partial infrastructure already in place** (discovered 2026-04-21):
  - `config/ns_teacher.yaml` — VP diffusion config (2 channels Vx/Vy, 128×128)
  - `src/utils/dataloader.py` `get_ns_loader` — reads PDEBench HDF5 layout
  - `loader_type: "ns"` wired through main loader
- **Nothing trained yet** — no checkpoints, no numbers on NS
- Blocking items:
  1. Confirm `data/2D_NS_incom_inhom_Re10000_128.h5` exists on server (else download from PDEBench)
  2. Train NS teacher (~1 day, 1 GPU)
  3. Train NS RF + Reflow + MFM in parallel with teacher (all teacher-independent)
  4. Train NS CD students (4/8/16) after teacher done (parallel on 3 GPUs)
  5. Train NS MM variants (pinned to struct-diversity peak per the Darcy lesson)
  6. Lock NS test indices → `data/ns_test_indices.npy`
  7. Extend `evaluate_paper.py` to handle 2-channel data (currently hardcoded 1-channel)
  8. Run unified eval on NS
- Estimated total: 3-4 days GPU time + ~1 day of code extension

## [MOMENT] Design spatially-aware moment loss
- Current moment loss aggregates over spatial dims before computing statistics,
  so it can't enforce spatial structure. Student satisfies it by producing
  varied-intensity samples with identical structure (see KNOWLEDGE.md).
- Candidate fixes:
  1. **Per-pixel moments**: match mean/var at each pixel location across batch
  2. **Patch-level moments**: divide image into patches, match moments per patch
  3. **Spatial power spectrum**: match 2D FFT power spectrum distribution
  4. **Explicit COM penalty**: add term that penalizes low center-of-mass variance
- If any of these work, promotes MM from "fix at epoch 75" to "fix throughout training"

## [RF] Investigate why Reflow underperforms RF
- Evaluated numbers: RF @ 5 steps WD 0.031, Reflow @ 5 steps WD 0.068
- Reflow should straighten trajectories and improve few-step quality — here it made things worse.
- Hypotheses to check:
  1. Reflow trained only 400 epochs (vs RF's 800) — undertrained?
  2. Issue in `reflow_pairs.pt` generation (paired data quality)
  3. RF trajectories were already straight enough that Reflow only added noise
- Diagnostics: re-sample Reflow at multiple checkpoints to see if quality improves with training, or regenerate the reflow pairs and retrain

## [MFM] Investigate MFM step-count insensitivity
- Evaluated numbers: WD ≈ 0.075 at 2, 4, 8, and 16 steps (flat)
- Normally more steps → better. MFM defies this.
- Hypotheses:
  1. MFM's averaged velocities saturate the quality early (by design)
  2. Numerical integration is dominated by model error, not step-count error
  3. MFM is underfit at exp_7 — the `r > t` loss may need more training
- Diagnostics: plot MFM loss curve, try longer training, test at 32/64 steps to see if it ever improves
