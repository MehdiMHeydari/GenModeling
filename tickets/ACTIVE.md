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

## Queued — CNS (PDEBench 2D Compressible Navier-Stokes, fourth-dataset benchmark)

> **Motivation.** Current paper has Darcy (radial spectrum, statistical
> proxy) + NS (∇·v=0, per-sample). Reviewer / PI concern: spectrum is
> "necessary but not sufficient" and structurally weaker than NS divergence
> (see TODO comment at top of methodology block in `ComSail paper/main.tex`).
> Adding a dataset with a categorically different per-sample physics check
> closes that gap. After a two-round cowork research sweep (papers + dataset
> collections), PDEBench 2D Compressible NS came out as the strongest pick:
> different physics (compressible, ∇·v ≠ 0), different check (pointwise
> positivity ρ>0, p>0 + integral conservation of mass/momentum/energy),
> drops into our existing PDEBench loader with minimal changes (same HDF5
> schema, native 128×128). Allen-Cahn from PDEGym is the alternative if we
> ever want maximum distance from fluids.

### [INFRA] Variant chosen
`2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5` (DaRUS file
ID 164690, ~51 GB). Reasoning:
- 128² native (no resampling, direct comparison to Darcy/NS)
- M=1.0 transonic (compressibility effects are meaningful; M=0.1 is
  nearly incompressible)
- η=ζ=0.01 (moderate viscosity = true compressible NS, not Euler;
  the 1e-08 variants are essentially inviscid)
- Rand IC + periodic BCs (standard PDEBench benchmark variant; periodic
  is what our `compute_pde_metrics.py` divergence op expects)
- Direct URL: `https://darus.uni-stuttgart.de/api/access/datafile/164690`

### [INFRA] BLOCKED: disk space
Tried `python scripts/PDEBench/download_direct.py --pde_name 2d_cfd` — case
mismatch bug in their script means uppercase input doesn't match the
lowercased CSV column. Lowercase works but pulls all 551 GB of 2D_CFD.
Direct DaRUS wget is the right approach.

Started direct wget into `~/GenModeling-main/data/`, **failed at 42 GB / 51 GB**
because `/ehome` filesystem is 100% full (4 TB shared across all users,
1.6 GB available when the wget hit the wall). Partial file
`~/GenModeling-main/data/2D_CFD_M1.0_128.h5` is still on disk at 42 GB.

Investigated alternative mounts on the GPU server (`comsail-amp.ame.nd.edu`):
- `/storage` (85 TB, 406 GB free) — **no write permission**
- `/home`   (64 TB, 752 GB free) — **no write permission**
- `/`       (877 GB, 814 GB free) — local OS disk, no user write access
- `/ehome`  (4 TB, 1.6 GB free) — the only writable mount, full

**Right long-term fix:** ask PI / cluster admin for write access to
`/storage/mehdi/`. 406 GB headroom there is the proper home for big
training data; `/ehome` being 100% full will eventually break checkpointing
on every training run we launch, not just this one.

### [INFRA] Short-term unblock: prune intermediate student checkpoints

Each Darcy/NS student exp saved a checkpoint every 25 epochs → 41
checkpoints × 403 MB ≈ 17 GB per exp, mostly unused.

**Safe pruning plan:** keep only every 100th checkpoint + the final one
(so per-exp we keep `checkpoint_0, _100, _200, ..., _900, _999` — 11
files, ~4.4 GB) and delete the in-between 25/75/125/175/... ones. Any
specific epoch can be recovered by retraining a few epochs from the
nearest 100-step checkpoint.

Dry-run to see what would be deleted (run first to confirm):
```
for d in ~/GenModeling-main/darcy_student/exp_* ~/GenModeling-main/ns_student/exp_*; do
    find "$d/saved_state" -maxdepth 1 -name 'checkpoint_*.pt' 2>/dev/null | \
    awk -F'[_.]' '$(NF-1) % 25 == 0 && $(NF-1) % 100 != 0 && $(NF-1) < 999 && $(NF-1) < 299 {print}'
done | head -30
```

Estimate total savings:
```
for d in ~/GenModeling-main/darcy_student/exp_* ~/GenModeling-main/ns_student/exp_*; do
    find "$d/saved_state" -maxdepth 1 -name 'checkpoint_*.pt' 2>/dev/null | \
    awk -F'[_.]' '$(NF-1) % 25 == 0 && $(NF-1) % 100 != 0 && $(NF-1) < 999 && $(NF-1) < 299 {print}'
done | xargs du -sh 2>/dev/null | awk '{sum += $1} END {print sum " GB roughly"}'
```

Actually delete (only when happy with the dry-run):
```
for d in ~/GenModeling-main/darcy_student/exp_* ~/GenModeling-main/ns_student/exp_*; do
    find "$d/saved_state" -maxdepth 1 -name 'checkpoint_*.pt' 2>/dev/null | \
    awk -F'[_.]' '$(NF-1) % 25 == 0 && $(NF-1) % 100 != 0 && $(NF-1) < 999 && $(NF-1) < 299 {print}'
done | xargs rm -v
```

Conservative estimate: ~150 GB freed across all student experiments.
Pruning the four 17-GB Darcy exps alone (exp_2, 3, 5, 21, 22) frees
~75 GB — more than enough to finish the CFD download + leave training
headroom.

### [INFRA] After cleanup: finish the CFD download
```
cd ~/GenModeling-main/data
wget -c -O 2D_CFD_M1.0_128.h5 "https://darus.uni-stuttgart.de/api/access/datafile/164690"
```
The `-c` flag resumes from the partial 42 GB file (~9 GB remaining,
~6 min at 25 MB/s).

Verify shape after download:
```
python -c "import h5py; f=h5py.File('data/2D_CFD_M1.0_128.h5','r'); print(list(f.keys())); [print(k, f[k].shape, f[k].dtype) for k in f.keys()]"
```

### [INFRA] Implementation TODO list (after data is on disk)

Files to create (all small, copy-edits of the NS equivalents):
- `scripts/preprocess_cns.py` — sanity-check 128² file, compute per-channel
  normalization stats (ρ, p, v_x, v_y all have different scales — must
  normalize per-channel, NOT globally like NS does)
- `scripts/lock_cns_test_indices.py` — 9000 train / 1000 test split
  (avoid the temporal-tail OOD issue flagged at the top of this file —
  use shuffled split for CNS from the start)
- `scripts/compute_pde_metrics.py` — add `cns_positivity(sample)` (per-pixel
  fraction of ρ≤0 or p≤0, plus min(ρ), min(p)) and `cns_conservation(sample)`
  (per-sample integrals of mass, momentum, energy). Reuse existing
  `ns_divergence` as a contrastive diagnostic — should read nonzero here.
- `src/eval/constants.py` — add `"cns"` entry to `_DATASETS`:
  `data_shape: (4, 128, 128)`, `data_path: "data/2D_CFD_M1.0_128.h5"`,
  `stats_dir: "cns_teacher/exp_1/saved_state"`, etc.
- `src/utils/dataloader.py` — add `get_cns_loader` (or extend
  `get_ns_loader` with a channel-flag) handling the 4-channel HDF5 with
  per-channel normalization
- Six configs under `config/`: `cns_teacher.yaml`, `cns_cm_cd.yaml`,
  `cns_cm_cd_prism.yaml`, `cns_rectified_flow.yaml`,
  `cns_rectified_flow_reflow.yaml`, `cns_mean_flow.yaml`. Each is a copy
  of the NS equivalent with `unet.dim: [4, 128, 128]` and the CNS data
  path.

### [INFRA] Parallel training plan once configs are ready

Same dependency DAG as the NS rollout:
- **Phase 1 (parallel, 3 GPUs):** Teacher (GPU 0), RF (GPU 1), MFM (GPU 2).
  Teacher / RF / MFM are mutually independent.
- **Phase 2 (parallel, 2 GPUs):** when Teacher finishes →
  `precompute_teacher_moments.py --dataset cns` (CPU/GPU, minutes),
  then CD (GPU 0) and PRISM (GPU 3).
- **Phase 3 (1 GPU):** when RF finishes →
  `generate_reflow_pairs.py --dataset cns` (hours), then Reflow (GPU 1).

Mirror the existing `scripts/run_ns_*.sh` chain scripts for CNS.

### [INFRA] Methodology gotchas to watch for
- **Per-channel normalization is essential.** ρ ∈ ~[0.5, 2], p ∈ ~[0.5, 2],
  velocities ∈ ~[-5, 5]. Globally min-max-normalizing all 4 channels
  together would compress the small-scale fields into nothing. The NS
  loader does global min-max — needs to be split per-channel for CNS.
- **Discretization floor for divergence is meaningless here.** The
  reference floor on CNS will be the *real-data positivity fraction*
  (should be 0% — every real sample is positive everywhere) and the
  *real-data distribution of conserved integrals*. Compute both before
  training so we have something to compare against.
- **Periodic BCs.** PDEBench's periodic CNS lets us reuse the central-
  difference div operator under the same BC assumption. Don't switch
  variants without re-checking.
- **Teacher canonical checkpoint TBD.** NS lessons: training is
  non-monotone; don't auto-pick the latest checkpoint. Plan a
  `diagnose_cns_teacher.py` sweep before locking the canonical ckpt.

### [PAPER] Cuts/edits to make once CNS is in
- Remove (or reconsider) the `% TODO(mehdi): consider cutting the radial
  spectrum` block in `ComSail paper/main.tex`. With CNS providing a third
  per-sample physics check on a different physics regime, the spectrum
  becomes "one of three checks" rather than "the only physics-aware
  check for Darcy," weakening the case to cut it.
- Update intro/abstract: "three datasets, three physics-aware checks
  (Darcy spectrum + NS divergence + CNS positivity/conservation)."
- Add a fourth column to the metrics heatmap (CNS panel).

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
