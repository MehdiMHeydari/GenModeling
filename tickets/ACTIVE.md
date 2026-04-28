# Active Work

> 🚨 **HIGH PRIORITY — NS test set must be redone before final paper results**
> The locked NS test indices (`data/ns_test_indices.npy`) are the LAST
> 1000 frames of the flattened time series. Sanity check (2026-04-27,
> `diagnostics/reflow_sanity/nn_distances.png`) showed those frames have
> NN distance ~52 to training, vs ~1 for random samples from full dataset.
> The held-out test set is structurally OOD from training — affects every
> NS pixel-level metric we have.
>
> **Required fix before paper submit:**
> 1. Redo train/test split with random shuffling, not contiguous tail
> 2. Retrain (training set will change → all checkpoints invalidated)
> 3. Rerun eval on new test set
>
> Or, if retrain is infeasible: document the test-set OOD-ness in a
> methodology footnote and rerun eval against random-from-full-dataset
> indices (acknowledging train-test contamination).
>
> Defer until after paper draft. Do not finalize numbers in paper until
> this is resolved.

> ⚠️ **wandb mixing note**: NS teacher, RF, MFM runs launched 2026-04-21/22
> log to the `darcy-*` wandb projects (see KNOWLEDGE.md). MM-mm21,
> MM-mm22, and any NS run launched after 2026-04-22 ~13:00 route to
> `ns-*` correctly.

## In flight — NS training (still running but eval already gives stable numbers)

### [INFRA] GPU 4: NS MFM (epoch ~725/1000)
- Tmux: `ns_mfm`
- Latest checkpoint included in current eval: `checkpoint_725.pt`
- Still improving with training but slowly; stable trend in eval

### [INFRA] GPU 5: NS MM-mm21 (epoch ~150/300)
- Tmux: `ns_moments`
- Latest in eval: ckpt 75 + ckpt 150
- Numbers flat past ckpt 75 — could kill if GPU is needed elsewhere

### [INFRA] GPU 1: NS MM-mm22 (epoch ~250/300)
- Tmux: `ns_mm22`
- Latest in eval: ckpt 75, 125, 250
- Same as mm21 — flat past peak, could kill

## Queued

### [PAPER] Generate NS sample grid + histograms
- `scripts/generate_ns_samples.py` (newly added 2026-04-27): pulls 8 samples per
  method from `config/ns_paper_eval.yaml`, side-by-side GT vs Gen velocity-
  magnitude grids
- Run after eval: `python scripts/generate_ns_samples.py --gpu 2`
- Visual sanity check that the Reflow @ 1 NFE win is real and not artifactual

### [PAPER] Fill in `paper/main.tex` with real numbers
- Two-dataset story now lockable: Darcy (Table N) + NS (Table M)
- Methods section can be drafted in parallel — doesn't depend on numbers

### [PAPER] Cross-dataset comparison plot
- Pareto: WD vs NFE, faceted by dataset, one curve per method
- Reflow line should sit at the bottom of the NS panel; RF line at the bottom of the Darcy panel
- Drives home "best method is dataset-dependent" → why we needed both datasets

### [DECISION] Should we kill MM jobs?
- Numbers are flat. ~15 days of GPU-hours each remaining.
- Kill saves time; cost is "we won't see if MM somehow improves at ckpt 300."
- Darcy says it doesn't. Recommend: kill.

## Cleanup

### [INFRA] Kill stale tmux sessions
Safe to kill: `eval_A`, `eval_B`, `eval_all`, `eval_mm`, `eval_rf`, `figures`,
`ns_diag`, `ns_download`, `ns_download_1`, `ns_rf`, `ns_teacher`,
`ns_teacher_cd`, `ns_teacher_diag2`, `ns_teacher_diag3`, `ns_eval`, `ns_eval2`,
`ns_reflow` (training finished at 399).
