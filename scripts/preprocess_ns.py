"""
Preprocess PDEBench NS_incom 2D data for this project.

Takes one or more raw `ns_incom_inhom_2d_512-N.h5` files, flattens the time
dimension, downsamples 512 -> 128 via 4x4 average pooling, randomly samples
10,000 frames (seeded), and writes a single merged HDF5 file with the same
`tensor` layout this project's loaders expect.

Run once per NS dataset setup. After running, point the NS teacher config at
the output file. The output is ~2 GB vs 18 GB of raw downloads.

Usage:
    python scripts/preprocess_ns.py \\
        --inputs data/ns_incom_inhom_2d_512-0.h5 data/ns_incom_inhom_2d_512-1.h5 \\
        --output data/ns_incom_128_merged.h5
"""

import argparse
import os

import h5py
import numpy as np

from src.eval.constants import SINGLE_SEED

TARGET_SAMPLES = 10_000  # match Darcy
TARGET_H = 128
TARGET_W = 128


def load_and_flatten(path):
    """Load a single NS file and return a flattened (N, C, H, W) float32 array."""
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print(f"  {os.path.basename(path)}  keys={keys}")
        if "velocity" in f:
            arr = np.array(f["velocity"])
        elif "Vx" in f and "Vy" in f:
            vx = np.array(f["Vx"])
            vy = np.array(f["Vy"])
            arr = np.stack([vx, vy], axis=-1)
        elif "tensor" in f:
            arr = np.array(f["tensor"])
        else:
            raise ValueError(f"Cannot find velocity data in {path}. Keys: {keys}")
    arr = arr.astype(np.float32)
    print(f"  raw shape: {arr.shape}")

    # PDEBench layout is typically [N_sim, T, H, W, V] (channels last) OR
    # [N_sim, T, V, H, W] (channels first, less common). Detect by axis sizes.
    if arr.ndim == 5:
        N, T, a, b, c = arr.shape
        # channels axis = smallest dim that's not H/W (H/W are 512)
        if c <= 4:  # channels last
            arr = arr.reshape(N * T, a, b, c).transpose(0, 3, 1, 2)  # -> (N*T, V, H, W)
        else:  # channels first
            arr = arr.reshape(N * T, a, b, c)  # already (N*T, V, H, W)
    elif arr.ndim == 4:
        # (N, H, W, V) or (N, V, H, W)
        if arr.shape[-1] <= 4:
            arr = arr.transpose(0, 3, 1, 2)
    elif arr.ndim == 3:
        arr = arr[:, np.newaxis]
    print(f"  flattened shape: {arr.shape}")
    return arr


def downsample_avg_pool(arr, factor_h, factor_w):
    """Average-pool spatial dims by an integer factor. arr: (N, C, H, W)."""
    N, C, H, W = arr.shape
    assert H % factor_h == 0 and W % factor_w == 0, \
        f"H={H}, W={W} must be divisible by factors ({factor_h}, {factor_w})"
    return arr.reshape(
        N, C, H // factor_h, factor_h, W // factor_w, factor_w
    ).mean(axis=(3, 5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="One or more raw NS hdf5 files")
    p.add_argument("--output", required=True)
    p.add_argument("--target_samples", type=int, default=TARGET_SAMPLES)
    p.add_argument("--target_h", type=int, default=TARGET_H)
    p.add_argument("--target_w", type=int, default=TARGET_W)
    args = p.parse_args()

    for path in args.inputs:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    # Load + flatten each file, concatenate
    chunks = []
    for path in args.inputs:
        print(f"Loading {path}")
        chunks.append(load_and_flatten(path))
    data = np.concatenate(chunks, axis=0)
    del chunks
    print(f"\nConcatenated shape: {data.shape}")

    # Downsample
    _, _, H, W = data.shape
    if (H, W) != (args.target_h, args.target_w):
        factor_h = H // args.target_h
        factor_w = W // args.target_w
        print(f"Downsampling {H}x{W} -> {args.target_h}x{args.target_w} "
              f"(avg pool {factor_h}x{factor_w})")
        data = downsample_avg_pool(data, factor_h, factor_w)
        print(f"Downsampled shape: {data.shape}")

    # Subsample to target count
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

    # Save with "tensor" key so existing loaders can pick it up
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("tensor", data=data, compression="gzip", compression_opts=4)
    print(f"\nWrote {args.output}")
    print(f"  shape: {data.shape}  dtype: {data.dtype}")
    print(f"  min/max: {data.min():.4f} / {data.max():.4f}")
    print(f"  size on disk: {os.path.getsize(args.output) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
