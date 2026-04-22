# Active Work

> ⚠️ **wandb mixing note**: NS teacher, RF, MFM runs launched 2026-04-21/22
> are logged in the `darcy-*` wandb projects (hardcoded project bug fixed
> in commit 2cc2d47 but not restarted). See KNOWLEDGE.md.
> MM-mm21, MM-mm22, and later NS runs route to `ns-*` correctly.

## In flight — NS training

### [INFRA] GPU 3: NS teacher (600 ep) → CD-16 (1000 ep) chain
- Script: `scripts/run_ns_teacher_then_cd.sh`
- Config: `config/ns_teacher.yaml` → `config/ns_cm_cd.yaml`
- Output: `ns_teacher/exp_1/saved_state/`, `ns_student/exp_1/saved_state/`
- Wandb (mixed): `darcy-teacher` for now
- Tmux: `ns_teacher_cd`

### [INFRA] GPU 0: NS RF round 1 (800 ep) → reflow pairs → reflow (400 ep)
- Script: `scripts/run_ns_rf_pipeline.sh`
- Configs: `config/ns_rectified_flow.yaml` → `config/ns_rectified_flow_reflow.yaml`
- Output: `ns_rectified_flow/exp_1/saved_state/`, `ns_rectified_flow_reflow/exp_1/saved_state/`
- Wandb (mixed): `darcy-rectified-flow`
- Tmux: `ns_rf`

### [INFRA] GPU 4: NS MFM (1000 ep, batch 8 due to GPU mem)
- Config: `config/ns_mean_flow.yaml` (batch_size dropped 16 → 8 reactively due to OOM on shared GPU)
- Output: `ns_mean_flow/exp_1/saved_state/`
- Wandb (mixed): `darcy-mean-flow`
- Tmux: `ns_mfm`
- Note: batch size reduction — call out in paper methods

### [INFRA] GPU 5: NS MM-mm21 (mu=4, var=200, 300 ep)
- Config: `config/ns_cm_cd_mm21.yaml`
- Teacher checkpoint: `ns_teacher/exp_1/saved_state/checkpoint_75.pt` (early checkpoint, may need restart later)
- Teacher moments: `ns_teacher/exp_1/saved_state/teacher_moments.pt` (precomputed from checkpoint_75)
- Output: `ns_student/exp_21/saved_state/`
- Wandb (correct): `ns-student`
- Tmux: `ns_mm21`

### [INFRA] GPU 1: NS MM-mm22 (mu=16, var=150, 300 ep)
- Config: `config/ns_cm_cd_mm22.yaml`
- Same teacher checkpoint + moments as MM-mm21
- Output: `ns_student/exp_22/saved_state/`
- Wandb (correct): `ns-student`
- Tmux: `ns_mm22`

## Queued (after training finishes)

### [EVAL] Extend pipeline to 2-channel data
- `src/eval/constants.py` has `DATA_SHAPE = (1, 128, 128)` hardcoded — needs parametrization
- `evaluate_paper.py`, `generate_paper_figures.py` also assume single-channel
- `structural_diversity` metric: decide between per-channel-then-average vs magnitude field `sqrt(Vx^2 + Vy^2)`
- Required before running NS eval

### [EVAL] Lock NS test indices
- `data/ns_test_indices.npy` with 1000 indices from the 8000-frame pool (train 7000 / test 1000)
- One-off variant of `scripts/lock_test_indices.py` for NS

### [EVAL] Run NS numeric eval
- Create `config/ns_paper_eval.yaml` listing every NS method
- Run `scripts/evaluate_paper.py --config config/ns_paper_eval.yaml`
- Output: `results/ns_eval_all.csv`

### [PAPER] Generate NS sample grid + histograms
- Extend `scripts/generate_paper_figures.py` for 2-channel NS (plot magnitude field, or Vx/Vy separately)

### [PAPER] Fill in `paper/main.tex` Methods section
- Can be drafted now; doesn't depend on numbers

### [DECISION] Restart NS MM on better teacher checkpoint?
- MM jobs are running against checkpoint_75 which is very early
- If teacher converges well past epoch 75, may want to restart MM on that checkpoint with fresh moments
- Decide after teacher training finishes and we see what checkpoint quality looks like

## Darcy figures — pending review

### [PAPER] Review `results/figures/sample_grid.png` + `histograms.png`
- Generated with MM pinned to checkpoint_75
- Pull: `scp -i ~/.ssh/comsailkey -r mehdi@comsail-amp.ame.nd.edu:~/GenModeling-main/results/figures ~/Downloads/`
