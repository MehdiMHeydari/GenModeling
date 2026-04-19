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

## [INFRA] Extend to additional PDE datasets
- User-requested: expand beyond Darcy Flow
- Candidates: Navier-Stokes (incompressible), Burgers, reaction-diffusion
- Use PDEBench data to reduce provenance issues
- Deferred until Darcy results are locked
