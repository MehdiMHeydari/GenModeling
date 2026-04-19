"""
One-time script to lock the held-out Darcy test indices for the paper.

Writes `data/test_indices.npy` with N_TEST_SAMPLES deterministic indices
drawn from the tail of the dataset (so they don't overlap with the first
9000 training samples). All paper-facing eval scripts load these indices
via `src.eval.constants.load_test_indices()`.

Run ONCE. Changing the output invalidates every prior eval run.

Usage:
    python scripts/lock_test_indices.py
"""

import os
import h5py
import numpy as np

from src.eval.constants import (
    DATA_PATH, TEST_INDICES_PATH, N_TEST_SAMPLES, SINGLE_SEED,
)


TRAIN_SAMPLES = 9000  # matches teacher config's dataloader.train_samples


def main():
    if os.path.exists(TEST_INDICES_PATH):
        raise RuntimeError(
            f"{TEST_INDICES_PATH} already exists. "
            f"Delete it manually if you really want to regenerate "
            f"(this will invalidate all prior eval runs)."
        )

    with h5py.File(DATA_PATH, "r") as f:
        total = f["tensor"].shape[0]

    if total <= TRAIN_SAMPLES:
        raise RuntimeError(
            f"Dataset has only {total} samples, need > {TRAIN_SAMPLES} for a held-out set."
        )

    held_out = np.arange(TRAIN_SAMPLES, total)
    if len(held_out) < N_TEST_SAMPLES:
        raise RuntimeError(
            f"Only {len(held_out)} held-out samples available, "
            f"need {N_TEST_SAMPLES}."
        )

    rng = np.random.default_rng(SINGLE_SEED)
    indices = np.sort(rng.choice(held_out, size=N_TEST_SAMPLES, replace=False))

    os.makedirs(os.path.dirname(TEST_INDICES_PATH), exist_ok=True)
    np.save(TEST_INDICES_PATH, indices)

    print(f"Locked {N_TEST_SAMPLES} test indices to {TEST_INDICES_PATH}")
    print(f"  range: [{indices.min()}, {indices.max()}]")
    print(f"  first 5: {indices[:5].tolist()}")
    print(f"  last 5:  {indices[-5:].tolist()}")


if __name__ == "__main__":
    main()
