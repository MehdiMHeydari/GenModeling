# GenModeling Project — Complete Reference

Single document with everything needed to understand, run, extend, or debug
this project. Written for an LLM/collaborator joining cold; assumes only
basic Python + ML background.

---

## 1. Project goal

Benchmark generative models as **few-step PDE surrogates**. The question:
given a probability distribution over physical fields (porosity, fluid
velocity, chemical concentrations), can we generate diverse, physically
plausible samples in 1-16 sampling steps instead of the 75-250 steps a
diffusion teacher needs?

Methods compared:
- **Teacher**: VP diffusion + DDIM sampling (the gold-standard reference)
- **CD** (Consistency Distillation): few-step student distilled from teacher
- **MM** (Moment Matching): CD + moment-of-sampled-batch loss to fix mode collapse
- **PD** (Progressive Distillation): step count halved each round
- **RF** (Rectified Flow): direct-trained ODE flow on raw data
- **Reflow**: round 2 of RF, retrained on RF round 1's deterministic couplings
- **MFM** (Mean Flow Matching): direct-trained with dual time embedding (t, r)

**Main contribution claim:** Moment Matching is a real fix for CD's
structural mode collapse. Validated on Darcy (88% → 95% of teacher
diversity on NS, 19% → 67% on Darcy).

**Secondary finding:** Best few-step method is dataset-dependent. RF wins on
Darcy, Reflow wins on NS. This is why the paper needs ≥2 datasets.

---

## 2. Datasets

| | Darcy Flow | Navier-Stokes 2D | Reaction-Diffusion 2D |
|---|---|---|---|
| Source | PDEBench `2D_DarcyFlow_beta1.0_Train.hdf5` | PDEBench `ns_incom_inhom_2d_512` | PDEBench `2D_diff-react_NA_NA.h5` |
| Physics | Pressure through porous medium | Incompressible NS | FitzHugh-Nagumo activator-inhibitor |
| Channels | 1 (porosity scalar) | 2 (Vx, Vy velocity) | 2 (u, v concentrations) |
| Resolution | 128×128 | 128×128 (downsampled 4× from 512) | 128×128 (already) |
| Train / test | 9000 / 1000 | 7000 / 1000 | 9000 / 1000 |
| Frame correlation | IID samples | Time series (frames temporally correlated) | Time series within each sim |
| Data file | `data/2D_DarcyFlow_beta1.0_Train.hdf5` | `data/ns_incom_128_merged.h5` | `data/rd_128_merged.h5` |
| Test indices | `data/test_indices.npy` | `data/ns_test_indices.npy` | `data/rd_test_indices.npy` (TODO) |

**Pre-processing scripts:**
- Darcy uses PDEBench file directly (no preprocessing).
- NS: `scripts/preprocess_ns.py` — flattens time, downsamples 512→128 via 4×4 avg pool, subsamples 10k frames.
- RD: `scripts/preprocess_rd.py` — walks nested PDEBench groups (each sim is `/XXXX/data` of shape `[T, H, W, V]`), flattens, transposes channels-first, subsamples 10k frames.

**Data shapes after preprocessing:**
- All h5 files have a single `tensor` key of shape `(N, C, 128, 128)` (channels-first).
- C=1 for Darcy, C=2 for NS and RD.

**Normalization:** min-max scale to `[-1, 1]` using stats stored in
`<dataset>_teacher/exp_1/saved_state/data_{min,max}.npy`. Stats are computed
on the training set and saved by the teacher dataloader on first epoch.

---

## 3. Architecture

**Single UNet architecture used by every method.** Defined per-config:

```yaml
unet:
  dim: [C, 128, 128]      # C = 1 (Darcy) or 2 (NS/RD)
  channel_mult: "1, 2, 4, 4"
  num_channels: 64
  num_res_blocks: 2
  num_head_channels: 32
  attention_resolutions: "32"
  use_new_attention_order: True
  use_scale_shift_norm: True
  class_cond: False
```

MFM adds `use_future_time_emb: True` for the (t, r) dual-time embedding.

Source: `src/models/networks/unet/unet.py:UNetModelWrapper` (imported as `UNetModel`).

---

## 4. Key files and what they do

### Configs (`config/`)
- `<dataset>_teacher.yaml` — VP diffusion teacher
- `<dataset>_rectified_flow.yaml` — RF round 1
- `<dataset>_rectified_flow_reflow.yaml` — Reflow round 2 (Darcy & NS only — RD TODO)
- `<dataset>_mean_flow.yaml` — MFM
- `<dataset>_cm_cd.yaml` — CD baseline
- `<dataset>_cm_cd_4step.yaml`, `*_8step.yaml` — CD students at different student_steps
- `<dataset>_cm_cd_mm21.yaml`, `*_mm22.yaml` — Moment Matching variants
- `paper_eval.yaml` — Darcy unified eval methods list (top-level `dataset: darcy`)
- `ns_paper_eval.yaml` — NS unified eval methods list (`dataset: ns`)

### Training scripts (`scripts/`)
- `train_vp_diffusion.py` — teacher
- `train_rectified_flow.py` — RF and Reflow (script differentiates by config)
- `train_mean_flow.py` — MFM
- `train_cm.py` — CD students (incl. MM variants — config flag activates moment loss)
- `train_pd.py` — Progressive Distillation (Darcy only)

### Pre-processing
- `scripts/preprocess_ns.py` — NS data
- `scripts/preprocess_rd.py` — RD data
- `scripts/lock_test_indices.py` — Darcy test indices
- `scripts/lock_ns_test_indices.py` — NS test indices

### Pre-computed artifacts
- `scripts/precompute_teacher_moments.py` — Darcy teacher moments (for MM)
- `scripts/precompute_ns_teacher_moments.py` — NS teacher moments (for MM)
- `scripts/generate_reflow_pairs.py` — RF round 1 → reflow training pairs

### Evaluation
- `scripts/evaluate_paper.py` — unified metrics across all methods (`--dataset ns|darcy`)
- `scripts/sweep_mm_checkpoints.py` — MM checkpoint sweep
- `scripts/diagnose_teacher.py` — Darcy teacher diagnostic
- `scripts/diagnose_ns_teacher.py` — NS teacher diagnostic
- `scripts/diagnose_rd_teacher.py` — RD teacher diagnostic
- `scripts/sanity_check_reflow.py` — Reflow result validation (NN, divergence, WD, interpolation)
- `scripts/check_ns_test_distribution.py` — verifies train/test |v| match

### Figure generation
- `scripts/generate_ns_samples.py` — NS sample grid for paper
- `scripts/generate_darcy_samples.py` — Darcy sample grid
- `scripts/generate_ns_cd_samples.py`, `generate_darcy_cd_samples.py` — CD-focused variants
- `scripts/generate_histograms.py` — histogram panels + overlays for both datasets
- `scripts/summarize_ns_diag.py`, `combine_ns_samples.py` — tile per-method PNGs

### Eval helpers (`src/eval/`)
- `constants.py` — locked test indices paths, paper seeds (0, 1, 2), N_TEST_SAMPLES (1000), `_DATASETS` dict with per-dataset paths/shapes/canonical-teacher-ckpt

---

## 5. Pipeline (full)

For each new dataset:

```
1. Download raw data → data/<file>.h5
2. Preprocess if needed → data/<dataset>_128_merged.h5
3. Lock test indices: scripts/lock_<dataset>_test_indices.py
4. Train teacher: train_vp_diffusion.py config/<dataset>_teacher.yaml  (~14h)
5. Run teacher diagnostic: diagnose_<dataset>_teacher.py
   → pick canonical checkpoint + DDIM step count
   → update src/eval/constants.py _DATASETS[dataset]
6. Pre-compute teacher moments (for MM): precompute_<dataset>_teacher_moments.py
7. Launch downstream training (parallel):
   - RF round 1                  (teacher-independent, ~21h)
   - MFM                         (teacher-independent, ~14h)
   - CD students                 (teacher-dependent, ~33h each)
   - MM variants                 (teacher-dependent, slow due to moment_every=1, days)
   After RF round 1:
   - generate_reflow_pairs.py
   - Reflow round 2              (~10h)
8. Evaluate: evaluate_paper.py --config config/<dataset>_paper_eval.yaml
9. Generate figures: generate_<dataset>_samples.py + generate_histograms.py
```

---

## 6. Eval methodology (locked)

From `src/eval/constants.py`:

```python
N_TEST_SAMPLES = 1000        # all available held-out samples
PAPER_SEEDS = (0, 1, 2)      # 3 seeds → mean ± std
SINGLE_SEED = 0              # for one-shot artifacts (test indices, moments)
SCHEDULE_S = 0.008           # cosine noise schedule offset

# Per-dataset canonical config:
_DATASETS["darcy"]:
  teacher_ckpt = "darcy_teacher/exp_1/saved_state/checkpoint_200.pt"
  teacher_ddim_steps = 250
_DATASETS["ns"]:
  teacher_ckpt = "ns_teacher/exp_1/saved_state/checkpoint_75.pt"
  teacher_ddim_steps = 75
_DATASETS["rd"]:
  teacher_ckpt = "rd_teacher/exp_1/saved_state/checkpoint_75.pt"
  teacher_ddim_steps = 75
```

**Metrics computed by evaluate_paper.py per (method, step_count, seed):**
- `pixel_mse` — element-wise MSE of unpaired gen vs real samples
- `wasserstein` — 1-Wasserstein on the scalar field (porosity for Darcy, |v| for NS, sqrt(u²+v²) for RD)
- `mean_err`, `std_err` — moment errors on raw arrays
- `skew_err`, `kurt_err` — higher moments on scalar field
- `pix_diversity` — mean pairwise L2 over flattened samples
- `struct_diversity` — mean pairwise L2 of per-sample center-of-mass on scalar field (catches mode collapse)
- `wall_clock_s` — sampling time

**Critical metric for paper: `struct_diversity`.** 1D Wasserstein and pix_diversity can both be fooled by "same-shape-different-intensity" mode collapse. struct_diversity catches it.

**Sampler interfaces (commit 67aed0e bug fix):**
- `RectifiedFlowSampler.sample(z, num_steps=N)` — kwarg `num_steps`
- `MeanSampler.sample(z, t_span_kwargs={"start":0, "end":1, "steps":N+1})` — kwarg `t_span_kwargs`
- `MultistepCMSampler.sample(z)` — no kwarg, step count baked into `model.student_steps`

Passing the wrong kwarg silently uses 100 default steps. Always verify NFE in eval CSV matches expectation.

---

## 7. Headline findings

### Darcy benchmark (1000 samples × 3 seeds)

| Method | Best NFE | Wasserstein | struct_div | % of teacher diversity |
|---|---|---|---|---|
| Teacher | 250 | 0.023 | 9.99 | 100% |
| **RF (5 step)** | **5** | **0.031** | 7.25 | 73% |
| MM-exp22 | 16 | 0.043 | 4.33 | 43% |
| MM-exp21 | 16 | 0.060 | 6.68 | 67% |
| Reflow @ any | 1-10 | ~0.068 | 10.13 | 102% |
| MFM | 2-16 | ~0.075 | 12.27 | 123% |
| **CD-16 (baseline)** | 16 | 0.126 | 2.87 | **29% (collapsed)** |
| CD-8 | 8 | 0.231 | 1.45 | 14% |
| CD-4 | 4 | 0.282 | 0.91 | **9% (catastrophic)** |

### NS benchmark (1000 samples × 3 seeds)

| Method | NFE | Wasserstein | struct_div | % of teacher diversity |
|---|---|---|---|---|
| **Reflow @ 1 step** | **1** | **0.0102** | 8.85 | **113% (more than teacher)** |
| RF @ 10 step | 10 | 0.016 | 8.89 | 113% |
| CD-16 ckpt 100 | 16 | 0.026 | 6.95 | 88% |
| Teacher | 75 | 0.028 | 7.85 | 100% |
| MFM @ 16 step | 16 | 0.034 | 8.47 | 108% |
| RF @ 5 step | 5 | 0.041 | 8.87 | 113% |
| MM-mm21 ckpt 75 | 16 | 0.081 | 7.46 | 95% |
| MM-mm22 ckpt 75 | 16 | 0.081 | 7.41 | 94% |

### RD benchmark
TODO — teacher just trained, downstream training queued.

### Cross-dataset insights
1. **Best method is dataset-dependent.** RF wins Darcy, Reflow wins NS. Without 2 datasets, the paper conclusion would be wrong.
2. **CD on NS is *not* catastrophic** (88% diversity vs Darcy's 9-29%). MM still helps but the lift is smaller.
3. **MM trade-off (consistent both datasets):** better diversity, worse Wasserstein than CD baseline. It's a diversity intervention, not a quality intervention.
4. **Reflow @ 1 step beats teacher on NS** (WD 0.010 vs 0.028, struct_div 8.85 vs 7.85). At 75× cheaper sampling. This is the headline numerical result.
5. **Reflow plateaus on Darcy** at WD ~0.068 regardless of step count, while RF improves with steps. The reflow trick worked on NS but not on Darcy — likely because Darcy's RF round 1 trajectories were already nearly straight.

---

## 8. Known bugs and gotchas

### Wasserstein cap-bias bug (commit eaa24c5, 2026-04-26)
`compute_metrics` capped the flattened (N, H, W) array at 100k by taking the
**first** 100k. Sample-major flattening means the first 100k pixels live in
the first ~6 frames. With NS sorted test indices, those 6 frames are
temporally clustered → narrow |v| distribution → WD inflated 10-30× across
all methods. **Fix:** random subsample with `np.random.RandomState(0)`.
Reproducible and unbiased.

### NS test set is OOD from training (HIGH PRIORITY for paper)
NS data is a flattened time series. The locked test indices (`np.arange(7000, 8000)` random subset) are the contiguous *tail* of the time series — they sit in a different flow regime from training. Sanity check (`diagnostics/reflow_sanity/nn_distances.png`): held-out test has NN-to-training ~52, while random-from-full ~1.

This affects every NS pixel-level metric. Marginal-distribution metrics (WD on |v|) are robust because train/test |v| histograms match (WD 0.018), but pixel-level metrics (NN, MSE) are inflated for everyone.

**Fix needed before paper submit:**
1. Redo train/test split with random shuffling instead of contiguous tail
2. Retrain (training set will change → all checkpoints invalidated)
3. Rerun eval

Or document as a methodology footnote and rerun eval against random-from-full GT (acknowledging train-test contamination on NN metric).

### NS teacher training is non-monotone
WD across epochs: 50 → 0.011, 75 → 0.014, **100 → 0.111**, 200 → 0.033, 400 → 0.087, 599 → 0.041. Likely an LR-schedule × raw-weights interaction.

**Lesson:** for NS, evaluate a *spread* of checkpoints per method, not just the latest. Same pattern observed on RD teacher.

**Canonical NS teacher: ckpt 75 + DDIM 75.** Locked.

### Sampler interface kwarg silently defaults to 100 steps (commit 67aed0e)
See section 6. Always check NFE in CSV.

### MM training is 12× slower than vanilla CD
`moment_every: 1` runs the full sampling chain every iteration. Setting `moment_every: 8` is 8× faster; we never tested it because Darcy lessons said MM is converged at ckpt 75 anyway.

### Wandb project mixing (NS runs launched 2026-04-21/22 before commit 2cc2d47)
First-wave NS training runs (teacher, RF round 1, MFM) log to `darcy-*` wandb projects because `train_*.py` had hardcoded project strings. Fixed at commit 2cc2d47 (project derived from `loader_type`). Earlier runs were not relaunched to preserve training progress.

For paper writeup, NS training curves come from BOTH `darcy-teacher`, `darcy-rectified-flow`, `darcy-mean-flow` (early NS runs) and `ns-*` projects (MM and later runs).

### MultistepCDLoss had hardcoded 1-channel sample shape (commit 0b67c65)
`sample_moment_loss` created `torch.randn(B, 1, 128, 128)` regardless of model channels. Broke NS MM training. Fixed: added `sample_shape` kwarg, passed `tuple(config.unet.dim)` from `train_cm.py`.

### generate_reflow_pairs hardcoded 1-channel UNet (commit e7eaa3f)
Same bug class. Fixed: added `--channels` CLI arg.

### NS RF round 1 was wiped during reflow restart (~April 23)
The pipeline script `run_ns_rf_pipeline.sh` re-ran RF round 1 from scratch instead of skipping. Fix: added `if [ -f "$ROUND1_FINAL_CKPT" ]; then skip` guard.

### `--only` flag in eval scripts uses `name@steps` format
e.g. `python scripts/evaluate_paper.py --only "Teacher@250,CD-4step@4"`.
Splits on `,` and `@`. Names must match config entries exactly.

### CSV header back-compat
`evaluate_paper.py` errors out if the output CSV has a different header
than the current one. Forces explicit decision when CSV format changes
(prevents silent appends with mismatched columns).

---

## 9. Operational

### GPU server access
- Host: `mehdi@comsail-amp.ame.nd.edu`
- SSH: `ssh -i /Users/mehdi/.ssh/comsailkey mehdi@comsail-amp.ame.nd.edu`
- SCP: `scp -i /Users/mehdi/.ssh/comsailkey ...`
- Workdir: `~/GenModeling-main`
- Conda env: `gen-modeling`
- Multi-tenant: GPUs 0-7 shared with other users; ours typically have 20-40 GB usage

### Tmux convention
Sessions named after the job: `ns_teacher`, `ns_rf`, `rd_teacher`, etc.
- Attach: `tmux attach -t <name>`
- Detach: Ctrl-B D
- Capture output: `tmux capture-pane -t <name> -p | tail -N`
- Kill: `tmux kill-session -t <name>`

### Wandb projects
- Darcy: `darcy-teacher`, `darcy-rectified-flow`, `darcy-mean-flow`, `darcy-student`, `darcy-pd`
- NS: `ns-teacher`, `ns-rectified-flow`, `ns-mean-flow`, `ns-student`
- RD: `rd-teacher`, `rd-rectified-flow`, `rd-mean-flow`
- (NS first-wave runs are in `darcy-*` due to bug — see section 8)

### Common SSH commands
```bash
# Status check
ssh ... "tmux ls && nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"

# Latest checkpoint per training dir
ssh ... "for d in \$(find ns_* rd_* darcy_* -type d -name saved_state 2>/dev/null); do
  echo === \$d ===
  ls \$d/checkpoint_*.pt 2>/dev/null | sed 's/.*checkpoint_//; s/\.pt//' | sort -n | tail -1
done"

# Pull a single file
scp -i /Users/mehdi/.ssh/comsailkey \
    mehdi@comsail-amp.ame.nd.edu:~/GenModeling-main/<path> \
    ~/Downloads/

# Pull a directory
scp -i /Users/mehdi/.ssh/comsailkey -r \
    mehdi@comsail-amp.ame.nd.edu:~/GenModeling-main/<dir> \
    ~/Downloads/
```

### User preferences (durable, do not violate)
- **Always push after committing** — user runs on remote GPU server, needs changes available
- **Never add Co-Authored-By trailers** to commits; never attribute Claude in any commit message
- **Tmux setup commands** should be multi-line, not one-liners with `; read` patterns
- Repository pushes to `https://github.com/MehdiMHeydari/GenModeling.git`

---

## 10. Current status (snapshot of 2026-04-29)

### Done
- Darcy full benchmark: Teacher, CD-4/8/16, MM-exp20/21/22, RF, Reflow, MFM, PD
- NS benchmark: Teacher, CD-16, CD-4 (epoch 999), RF, Reflow, MFM (epoch 999), MM-mm21 (299), MM-mm22 (299)
- NS missing: CD-8 (never trained)
- Paper figures: sample grids, histograms, CD-focused for both datasets
- RD: data preprocessed, teacher trained (epoch 599), diagnostic done, canonical = ckpt 75 + DDIM 75
- Reflow sanity check (3 of 4 tests passed; SLERP test passed on re-run)

### In flight (post-RD-teacher-decision)
- RD RF round 1 (queued for GPU 3)
- RD MFM (queued for GPU 5)
- RD CD-16 baseline (waiting for teacher_moments.pt)
- RD MM variant (waiting for teacher_moments.pt)
- RD Reflow round 2 (waits on RF round 1)
- RD test indices lock (scripts/lock_rd_test_indices.py — TODO)

### Open paper-blockers
- 🚨 NS test set OOD must be fixed (see section 8)
- Re-run NS eval to add CD-4 + bump MFM/MM final checkpoints
- RD downstream training (~5 days GPU time)
- RD eval and figures
- Cross-dataset Pareto plot
- Paper text

---

## 11. Useful one-liners

```bash
# Convert to JSON: read all eval CSVs into a single dataframe-friendly format
python -c "import pandas as pd; print(pd.concat([pd.read_csv(f) for f in ['results/eval_all.csv', 'results/ns_eval_all.csv']]).to_csv())"

# Show all method names from a config
python -c "import yaml; print('\n'.join(m['name'] for m in yaml.safe_load(open('config/ns_paper_eval.yaml'))['methods']))"

# Quickly inspect any h5 file
python -c "import h5py; f=h5py.File('data/rd_128_merged.h5'); print({k: f[k].shape for k in f.keys()})"

# Check a teacher checkpoint loads cleanly
python -c "
import torch
state = torch.load('rd_teacher/exp_1/saved_state/checkpoint_75.pt', map_location='cpu', weights_only=True)
print({k: v.shape for k,v in state.items() if torch.is_tensor(v)})
"

# Print all experiment dirs and how many checkpoints each has
for d in $(find . -type d -name saved_state); do
  count=$(ls $d/checkpoint_*.pt 2>/dev/null | wc -l)
  echo "$d  ($count checkpoints)"
done
```
