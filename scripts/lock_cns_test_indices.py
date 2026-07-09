"""
One-time script to lock the held-out CNS test indices.

Fixes the temporal-tail OOD issue flagged for NS at the top of ACTIVE.md
by drawing the test indices as a *random shuffled* subset of the full
dataset rather than the contiguous tail. Train indices = complement.
Both saved so the loader can filter train samples by exclusion.

Run ONCE. Changing the output invalidates every prior CNS eval run.

Usage:
    python scripts/lock_cns_test_indices.py
"""

import os
import h5py
import numpy as np

from src.eval.constants import N_TEST_SAMPLES, SINGLE_SEED, get_dataset


def main():
    ds = get_dataset("cns")
    out_path = ds["test_indices_path"]

    if os.path.exists(out_path):
        raise RuntimeError(
            f"{out_path} already exists. Delete it manually if you really "
            "want to regenerate (invalidates all prior CNS eval runs)."
        )

    with h5py.File(ds["data_path"], "r") as f:
        total = f["tensor"].shape[0]

    if total < N_TEST_SAMPLES + 1000:
        raise RuntimeError(
            f"Dataset has {total} samples, need at least {N_TEST_SAMPLES + 1000} "
            "for a reasonable train/test split."
        )

    rng = np.random.default_rng(SINGLE_SEED)
    all_idx = np.arange(total)
    rng.shuffle(all_idx)
    test_idx  = np.sort(all_idx[:N_TEST_SAMPLES])
    train_idx = np.sort(all_idx[N_TEST_SAMPLES:])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, test_idx)

    train_path = out_path.replace("test", "train")
    np.save(train_path, train_idx)

    print(f"Locked {N_TEST_SAMPLES} CNS test indices to {out_path}")
    print(f"  test  range: [{test_idx.min()}, {test_idx.max()}]")
    print(f"  test  first 5: {test_idx[:5].tolist()}")
    print(f"  train count: {len(train_idx)}, saved to {train_path}")


if __name__ == "__main__":
    main()
