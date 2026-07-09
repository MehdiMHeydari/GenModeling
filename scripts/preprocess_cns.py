"""
Preprocess PDEBench 2D Compressible Navier-Stokes data.

Takes the raw `2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
file (~52 GB, 10,000 trajectories x 21 timesteps x 128x128 x 4 fields) and
writes a single merged file with the flat single-snapshot layout the rest
of the pipeline expects.

Differences from preprocess_ns.py:
  - Four named channels (density, pressure, Vx, Vy) instead of one 'tensor'
    key. We stack them channels-first as (N, 4, H, W).
  - Per-channel normalization stats (density ~ [0.5, 2], velocities ~ [-5, 5]
    are very different scales -- global min-max would compress the small
    scale ranges to nothing).

Run once, then point the CNS teacher config at the output file.

Usage:
    python scripts/preprocess_cns.py \\
        --input  /ehome/mehdi/storage/2D_CFD_M1.0_128.h5 \\
        --output data/cns_128_merged.h5
"""

import argparse
import os

import h5py
import numpy as np

from src.eval.constants import SINGLE_SEED

TARGET_SAMPLES = 10_000  # match Darcy / NS
CHANNEL_ORDER  = ("density", "pressure", "Vx", "Vy")  # order in the output


def load_channels(path):
    """Load and stack the four PDEBench CNS channels."""
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print(f"  {os.path.basename(path)}  keys={keys}")
        arrs = []
        for name in CHANNEL_ORDER:
            if name not in f:
                raise ValueError(f"Missing '{name}' in {path}. Keys: {keys}")
            arr = np.array(f[name], dtype=np.float32)
            print(f"    {name} raw shape: {arr.shape}")
            arrs.append(arr)
    # each arr is (N_traj, T, H, W); stack on channel axis 1
    data = np.stack(arrs, axis=2)  # -> (N_traj, T, 4, H, W)
    N, T, C, H, W = data.shape
    # flatten (traj, time) into a single snapshot axis
    data = data.reshape(N * T, C, H, W)
    print(f"  flattened shape: {data.shape}  ({N} traj x {T} steps)")
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Path to raw PDEBench 2D CFD HDF5")
    p.add_argument("--output", required=True)
    p.add_argument("--target_samples", type=int, default=TARGET_SAMPLES)
    args = p.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    print(f"Loading {args.input}")
    data = load_channels(args.input)  # (N*T, 4, 128, 128)

    n_available = data.shape[0]
    if n_available < args.target_samples:
        print(f"WARNING: only {n_available} frames, wanted {args.target_samples}. "
              "Keeping all.")
        indices = np.arange(n_available)
    else:
        rng = np.random.default_rng(SINGLE_SEED)
        indices = np.sort(rng.choice(n_available, args.target_samples, replace=False))
    data = data[indices]
    print(f"Subsampled to {data.shape[0]} frames (seed={SINGLE_SEED})")

    # Per-channel min/max. Save both, print for sanity checking.
    per_channel_min = data.reshape(data.shape[0], data.shape[1], -1).min(axis=(0, 2))
    per_channel_max = data.reshape(data.shape[0], data.shape[1], -1).max(axis=(0, 2))
    print("Per-channel stats:")
    for i, name in enumerate(CHANNEL_ORDER):
        print(f"  {name:8s}  min={per_channel_min[i]:+.4f}  max={per_channel_max[i]:+.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("tensor", data=data, compression="gzip", compression_opts=4)
        f.create_dataset("channel_names",
                         data=np.array(CHANNEL_ORDER, dtype="S16"))
        f.create_dataset("per_channel_min", data=per_channel_min.astype(np.float32))
        f.create_dataset("per_channel_max", data=per_channel_max.astype(np.float32))
    print(f"\nWrote {args.output}")
    print(f"  shape: {data.shape}  dtype: {data.dtype}")
    print(f"  size on disk: {os.path.getsize(args.output) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
