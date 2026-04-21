# Active Work

## Work streams currently in flight

### [EVAL] Re-run unified Darcy eval with struct_diversity metric
- Running overnight on 1 GPU, single tmux session (`eval_all`)
- New metrics: `pix_diversity` + `struct_diversity` per row
- MM variants pinned to `checkpoint_75.pt` (struct-diversity peak)
- Output: `results/eval_all.csv`
- Status: 🟡 running

### [INFRA] Download NS data files
- Two tmux sessions on the server:
  - `ns_download`: `ns_incom_inhom_2d_512-0.h5` (fileId 133280, ~9.3 GB)
  - `ns_download_1`: `ns_incom_inhom_2d_512-1.h5` (fileId 136439, ~9.3 GB)
- Status: 🟡 running
- Expected total: ~18.6 GB, probably hours depending on bandwidth

## Queued (after downloads finish)

### [INFRA] Preprocess NS data
- `scripts/preprocess_ns.py --inputs ...-0.h5 ...-1.h5 --output data/ns_incom_128_merged.h5`
- Flattens time dim, downsamples 512→128 (4×4 avg pool), subsamples to 10k frames
- Output ~2 GB; raw 18.6 GB can be deleted after
- Blocks NS teacher training

### [INFRA] Train NS teacher
- `python scripts/train_vp_diffusion.py config/ns_teacher.yaml`
- Blocked on preprocessing
- ~1 day on 1 GPU
- NOTE: NS config has `dim: [2, 128, 128]` (2 channels, Vx/Vy) vs Darcy's `[1, 128, 128]`. Need to verify downstream scripts handle 2-channel correctly before launching full pipeline.

### [INFRA] Train NS teacher-independent methods in parallel
- RF, Reflow, MFM all train from ground truth, not from teacher
- Can train on separate GPUs while NS teacher is still training
- Configs don't exist yet — need to create NS-specific variants of each
- Each ~1 day on 1 GPU

## Queued (after Darcy eval finishes)

### [PAPER] Regenerate visual sample figures
- `scripts/generate_paper_figures.py` with pinned MM checkpoints
- Produces `sample_grid.png` + `histograms.png` consistent with new eval numbers

### [PAPER] Fill in `paper/main.tex` Methods section
- Methods prose is ready to write (doesn't depend on numbers)
- Experiments section waits for Darcy + NS numbers

## Longer-horizon (paper-blocking)

### [INFRA] Extend pipeline to 2-channel data
- `src/eval/constants.py` has `DATA_SHAPE = (1, 128, 128)` hardcoded — needs parametrization
- `evaluate_paper.py`, `generate_paper_figures.py` sample with that shape
- `structural_diversity` metric assumes single-channel in the COM calculation; should work per-channel then average, or concatenate channels
- Required before running NS eval

### [EVAL] Lock NS test indices
- `data/ns_test_indices.npy` with 1000 indices (analogous to Darcy)
- New one-time script or extend `lock_test_indices.py` to handle both datasets
