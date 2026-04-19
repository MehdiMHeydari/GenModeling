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

## [INFRA] Extend to additional PDE datasets (required before paper conclusions)
- User explicitly wants a second dataset before claiming "X method is better than Y"
- Candidates: Navier-Stokes (incompressible), Burgers, reaction-diffusion
- Use PDEBench data to reduce provenance issues
- Blocks the paper's final comparative claims

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
