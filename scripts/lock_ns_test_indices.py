"""
One-time script to lock the held-out NS test indices for the paper.

Writes `data/ns_test_indices.npy` with N_TEST_SAMPLES deterministic indices
drawn from the tail of the NS dataset (last 1000 frames, past the 7000
training frames). Paper-facing NS eval scripts load these via
`src.eval.constants.load_test_indices('ns')`.

Run ONCE. Changing the output invalidates every prior NS eval run.

Usage:
    python scripts/lock_ns_test_indices.py
"""

import os
import h5py
import numpy as np

from src.eval.constants import N_TEST_SAMPLES, SINGLE_SEED, get_dataset


def main():
    ds = get_dataset("ns")
    out_path = ds["test_indices_path"]

    if os.path.exists(out_path):
        raise RuntimeError(
            f"{out_path} already exists. "
            f"Delete it manually if you really want to regenerate "
            f"(this will invalidate all prior NS eval runs)."
        )

    with h5py.File(ds["data_path"], "r") as f:
        total = f["tensor"].shape[0]

    train_n = ds["train_samples"]
    if total <= train_n:
        raise RuntimeError(
            f"Dataset has only {total} samples, need > {train_n} for a held-out set."
        )

    held_out = np.arange(train_n, total)
    if len(held_out) < N_TEST_SAMPLES:
        raise RuntimeError(
            f"Only {len(held_out)} held-out samples available, "
            f"need {N_TEST_SAMPLES}."
        )

    rng = np.random.default_rng(SINGLE_SEED)
    indices = np.sort(rng.choice(held_out, size=N_TEST_SAMPLES, replace=False))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, indices)

    print(f"Locked {N_TEST_SAMPLES} NS test indices to {out_path}")
    print(f"  range: [{indices.min()}, {indices.max()}]")
    print(f"  first 5: {indices[:5].tolist()}")
    print(f"  last 5:  {indices[-5:].tolist()}")


if __name__ == "__main__":
    main()
