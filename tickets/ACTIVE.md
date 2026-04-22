# Active Work

## In flight — overnight NS training

### [INFRA] GPU 3: NS teacher (600 ep) → CD-16 (1000 ep) chain
- Script: `scripts/run_ns_teacher_then_cd.sh`
- Config: `config/ns_teacher.yaml` → `config/ns_cm_cd.yaml`
- Output dirs: `ns_teacher/exp_1/saved_state/`, `ns_student/exp_1/saved_state/`
- Runtime estimate: ~2-2.5 days (teacher + CD sequential on one GPU)
- Teacher checkpoints every 25 epochs (resumable via `restart: True, restart_epoch: N`)

### [INFRA] GPU 0: NS RF round 1 (800 ep) → reflow pairs → reflow (400 ep)
- Script: `scripts/run_ns_rf_pipeline.sh`
- Configs: `config/ns_rectified_flow.yaml` → `config/ns_rectified_flow_reflow.yaml`
- Output dirs: `ns_rectified_flow/exp_1/saved_state/`, `ns_rectified_flow_reflow/exp_1/saved_state/`
- Reflow pairs: `ns_rectified_flow/reflow_pairs.pt`
- Runtime estimate: ~1.5 days
- Teacher-independent — does not block on the GPU 3 teacher

### [INFRA] GPU 1: NS MFM (1000 ep)
- Config: `config/ns_mean_flow.yaml`
- Output dir: `ns_mean_flow/exp_1/saved_state/`
- Runtime estimate: ~1 day
- Teacher-independent — does not block on the GPU 3 teacher

### [EVAL] Darcy sample grid / histogram figures
- `scripts/generate_paper_figures.py` on a free GPU
- Output: `results/figures/sample_grid.png`, `results/figures/histograms.png`
- MM rows now at `checkpoint_75.pt` (diversity peak), not the late collapsed epochs
- Status: 🟡 running (or may have finished — check)

## Queued (after overnight training finishes)

### [EVAL] Run NS numeric eval
- Extend `src/eval/constants.py` / `evaluate_paper.py` for 2-channel data
- Lock `data/ns_test_indices.npy` (1000 held-out indices from the 8000 total, leaving 7000 train)
- Create `config/ns_paper_eval.yaml` listing every NS method
- Run `scripts/evaluate_paper.py` with NS config
- Produces `results/ns_eval_all.csv`

### [EVAL] Train NS Moment Matching variants
- Requires NS teacher + NS teacher_moments.pt precomputed
- Pick 1-2 hyperparameter settings (from Darcy lessons: mu=4 var=200 was struct-diversity peak)
- Each ~1 day of training

### [PAPER] Fill in `paper/main.tex` Methods section
- Can be drafted now; doesn't depend on numbers
- Use `EXPERIMENT_LOG.md` + session notes for content

### [PAPER] Generate NS sample grid + histograms
- Extend `scripts/generate_paper_figures.py` for 2-channel NS

## Longer-horizon (paper-blocking)

### [INFRA] Extend pipeline to 2-channel data
- `src/eval/constants.py` has `DATA_SHAPE = (1, 128, 128)` hardcoded — needs parametrization
- `evaluate_paper.py`, `generate_paper_figures.py` sample with that shape
- `structural_diversity` metric should work per-channel then average (or on the magnitude field `sqrt(Vx^2 + Vy^2)`)
- Required before running NS eval
