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

## In flight

**Nothing.** All NS training jobs have finished. RD teacher just finished
at epoch 599. GPUs 3, 5, 6, 7 are idle.

## Queued — RD downstream training (the third-dataset benchmark)

### [INFRA] RD teacher diagnostic (highest priority before downstream)
- Mirror `scripts/diagnose_ns_teacher.py` for RD; pick canonical
  checkpoint + step count
- NS taught us training is non-monotone — don't blindly use the
  latest checkpoint
- Sweep across {early, mid, late} ckpts × DDIM step counts
- ~30 min per checkpoint on a free GPU

### [INFRA] RD RF round 1 (teacher-independent, can launch now)
- Need: `config/rd_rectified_flow.yaml` (TODO — copy from NS)
- 800 epochs, ~21h on A100
- Output: `rd_rectified_flow/exp_1/saved_state/`

### [INFRA] RD Reflow round 2 (after RF round 1 finishes)
- Need: `config/rd_rectified_flow_reflow.yaml`
- ~10h after RF done

### [INFRA] RD MFM (teacher-independent, can launch now)
- Need: `config/rd_mean_flow.yaml`
- 1000 epochs, ~14h

### [INFRA] RD CD-16 baseline (teacher-dependent — wait for diagnostic)
- Need: `config/rd_cm_cd.yaml`
- ~33h

### [INFRA] RD MM variant (teacher-dependent — wait for diagnostic)
- Need: `config/rd_cm_cd_mm.yaml` + precomputed teacher moments
- Slow due to moment_every=1 — ~4 days

## Queued — NS

### [EVAL] Re-run NS eval with new checkpoints
- Add CD-4 (epoch 999) row + update MFM/MM rows to final checkpoints
- Update `config/ns_paper_eval.yaml` first (add CD-4 entry, bump MFM 725→999, bump MM 150/250→299)
- Run on a free GPU, ~3-4h
- Results CSV: append/replace in `results/ns_eval_all.csv`

## Queued — Paper writing

### [PAPER] Fill in `paper/main.tex` with real numbers
- Two-dataset story now lockable (Darcy + NS)
- RD will be a third panel once benchmark is in
- Methods section can be drafted now — doesn't depend on numbers

### [PAPER] Cross-dataset comparison plot
- Pareto: WD vs NFE, faceted by dataset
- "Best method is dataset-dependent" headline plot

## Cleanup

### [INFRA] Kill stale tmux sessions (pile-up since Apr 18)
Safe to kill (everything has either finished or is months stale):
`check`, `darcy_cd`, `eval_A`, `eval_B`, `eval_all`, `eval_mm`, `eval_rf`,
`figures`, `ns_cd4`, `ns_diag`, `ns_download`, `ns_download_1`, `ns_eval`,
`ns_eval2`, `ns_mfm`, `ns_mm22`, `ns_moments`, `ns_reflow`, `ns_rf`,
`ns_samples`, `ns_teacher`, `ns_teacher_cd`, `ns_teacher_diag2`,
`ns_teacher_diag3`, `rd_teacher` (training finished).
