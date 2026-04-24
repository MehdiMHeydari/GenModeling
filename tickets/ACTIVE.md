# Active Work

> ⚠️ **wandb mixing note**: NS teacher, RF, MFM runs launched 2026-04-21/22
> log to the `darcy-*` wandb projects (see KNOWLEDGE.md). MM-mm21,
> MM-mm22, and any NS run launched after 2026-04-22 ~13:00 route to
> `ns-*` correctly.

## In flight — NS training

### [INFRA] GPU 0: NS Reflow round 2 (epoch ~233/400)
- Tmux: `ns_reflow`
- ~80s/iter, ETA ~3-4 hours from 2026-04-24 morning
- Output: `ns_rectified_flow_reflow/exp_1/saved_state/`

### [INFRA] GPU 4: NS MFM (epoch ~375/1000)
- Tmux: `ns_mfm`
- Slow due to batch 8 + JVP (OOM at batch 16)
- Output: `ns_mean_flow/exp_1/saved_state/`

### [INFRA] GPU 5: NS MM-mm21 (mu=4, var=200) — VERY SLOW
- Tmux: `ns_moments`
- Epoch 75/300, ETA ~4 days at ~28 min/epoch
- Darcy diversity-peak analog = checkpoint 75, which is SAVED — eval-ready now
- Decision (2026-04-24): let it continue past 75 for trend check, kill if no improvement at 100

### [INFRA] GPU 1: NS MM-mm22 (mu=16, var=150)
- Tmux: `ns_mm22`
- Epoch 125/300, same pace
- Checkpoints 75 and 125 both available for eval

## Queued

### [EVAL] Run NS numeric eval
- All pipeline pieces landed 2026-04-24: `config/ns_paper_eval.yaml`,
  `scripts/lock_ns_test_indices.py`, `src/eval/constants.py` (NS block),
  `scripts/evaluate_paper.py --dataset ns`
- Run order after latest git pull:
  ```
  python scripts/lock_ns_test_indices.py
  python scripts/evaluate_paper.py --gpu <free> \
      --config config/ns_paper_eval.yaml \
      --output results/ns_eval_all.csv
  ```
- Checkpoint spread per method already in the config (see file comments)
- Re-run after Reflow/MFM finish to pick up the final checkpoints

### [PAPER] Generate NS sample grid + histograms
- Extend `scripts/generate_paper_figures.py` for 2-channel NS
- Reuse velocity-magnitude reduction already implemented in
  `diagnose_ns_teacher.py` / `evaluate_paper.py:to_scalar_field`

### [PAPER] Fill in `paper/main.tex` Methods section
- Can be drafted now; doesn't depend on numbers

### [DECISION] Restart NS MM on a different teacher checkpoint?
- Moments precomputed from teacher checkpoint_75
- **2026-04-24 diagnostic confirms ckpt 75 is the best NS teacher** (WD
  0.014, beats ckpt 200's 0.033 and ckpt 599's 0.041). So no restart
  needed — we got lucky.

## Cleanup

### [INFRA] Kill stale tmux sessions
Safe to kill once confirmed finished: `eval_A`, `eval_B`, `eval_all`,
`eval_mm`, `eval_rf`, `figures`, `ns_diag`, `ns_download`, `ns_download_1`,
`ns_rf`, `ns_teacher`, `ns_teacher_cd` (CD-16 finished), `ns_teacher_diag2`,
`ns_teacher_diag3`.
