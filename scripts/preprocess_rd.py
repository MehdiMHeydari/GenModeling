"""
Preprocess PDEBench 2D Reaction-Diffusion data for this project.

PDEBench's RD file (`2D_diff-react_NA_NA.h5`) is a nested-group HDF5:
  /0000/data    shape (T=101, H=128, W=128, V=2)   group per simulation
  /0001/data    ...
  /0002/data    ...
  ...

This script flattens all sim×timestep frames into a single `(N, 2, 128, 128)`
tensor, subsamples to 10,000 frames (seed-locked), and writes a single merged
HDF5 with the same `tensor` key our loader expects. Resolution is already 128
so no downsampling needed.

Usage:
    python scripts/preprocess_rd.py \\
        --input data/2D_diff-react_NA_NA.h5 \\
        --output data/rd_128_merged.h5
"""

import argparse
import os

import h5py
import numpy as np

from src.eval.constants import SINGLE_SEED


TARGET_SAMPLES = 10_000


def collect_frames(h5path):
    """Walk the nested groups and pull every per-sim 'data' tensor.
    Returns concatenated array of shape (N_frames, 2, 128, 128) in float32."""
    chunks = []
    with h5py.File(h5path, "r") as f:
        sim_keys = sorted(k for k in f.keys() if "data" in f[k])
        if not sim_keys:
            # alternative: maybe a flat 'data' dataset at top level
            if "data" in f:
                arr = np.array(f["data"])
                chunks.append(arr)
            else:
                raise ValueError(
                    f"No 'data' datasets found in {h5path}. Top-level keys: {list(f.keys())}")
        for k in sim_keys:
            arr = np.array(f[k]["data"])  # (T, H, W, V)
            chunks.append(arr)
    print(f"  found {len(chunks)} simulation chunks")
    raw = np.concatenate(chunks, axis=0)  # (N_sim*T, H, W, V) if T concatenated correctly
    # If chunks were per-simulation (T, H, W, V), concat on axis=0 stacks T's
    # along sim dimension correctly.
    print(f"  raw concatenated shape: {raw.shape}  dtype={raw.dtype}")
    if raw.ndim != 4:
        raise ValueError(f"Expected 4D (N, H, W, V), got {raw.shape}")
    # PDEBench is channels-last: (N, H, W, V) -> (N, V, H, W)
    arr = raw.transpose(0, 3, 1, 2).astype(np.float32)
    print(f"  after transpose: {arr.shape}")
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="PDEBench 2D_diff-react_NA_NA.h5")
    p.add_argument("--output", required=True)
    p.add_argument("--target_samples", type=int, default=TARGET_SAMPLES)
    args = p.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    print(f"Loading {args.input}")
    data = collect_frames(args.input)
    print(f"\nFlattened shape: {data.shape}")
    print(f"  channels: u={data[:, 0].mean():.4f}±{data[:, 0].std():.4f}  "
          f"v={data[:, 1].mean():.4f}±{data[:, 1].std():.4f}")

    n_available = data.shape[0]
    if n_available < args.target_samples:
        print(f"WARNING: only {n_available} frames available, wanted "
              f"{args.target_samples}. Keeping all.")
        indices = np.arange(n_available)
    else:
        rng = np.random.default_rng(SINGLE_SEED)
        indices = np.sort(rng.choice(n_available, args.target_samples, replace=False))
    data = data[indices]
    print(f"Subsampled to {data.shape[0]} frames (seed={SINGLE_SEED})")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("tensor", data=data, compression="gzip", compression_opts=4)
    print(f"\nWrote {args.output}")
    print(f"  shape: {data.shape}  dtype: {data.dtype}")
    print(f"  min/max: {data.min():.4f} / {data.max():.4f}")
    print(f"  size on disk: {os.path.getsize(args.output) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
