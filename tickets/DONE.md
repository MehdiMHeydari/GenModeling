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
