"""
Quick diagnostic: does the locked NS test set have the same |v| distribution
as the training set?

Hypothesis: NS data is a flattened time series, so contiguous frames are
correlated. If the test indices live in the temporal tail (last 1000 frames)
they may sit in a flow regime not represented in training, inflating
Wasserstein on |v| uniformly across all evaluated methods.

Output:
    diagnostics/ns_test_dist/v_hist.png  — overlaid |v| histograms
                                            (train / test / full)
    stdout                                — summary stats per slice

Usage:
    python scripts/check_ns_test_distribution.py
"""

import os

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

from src.eval.constants import get_dataset, load_test_indices


OUT_DIR = "diagnostics/ns_test_dist"


def magnitude(field):
    """field: (N, 2, H, W) -> (N, H, W) speed."""
    return np.sqrt(field[:, 0] ** 2 + field[:, 1] ** 2)


def summary(name, field):
    flat = field.flatten()
    print(f"  {name:20s}  n={len(flat):>9,}  "
          f"mean={flat.mean():.4f}  std={flat.std():.4f}  "
          f"min={flat.min():.4f}  max={flat.max():.4f}  "
          f"p50={np.percentile(flat, 50):.4f}  "
          f"p99={np.percentile(flat, 99):.4f}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = get_dataset("ns")
    train_n = ds["train_samples"]

    with h5py.File(ds["data_path"], "r") as f:
        data = f["tensor"][:]
    total = len(data)
    print(f"NS dataset: {total} frames, train_samples={train_n}, "
          f"shape per frame = {data.shape[1:]}")

    test_idx = load_test_indices("ns")
    print(f"Locked test indices: n={len(test_idx)}, "
          f"range=[{test_idx.min()}, {test_idx.max()}]")

    train_field = magnitude(data[:train_n])
    test_field = magnitude(data[test_idx])
    full_field = magnitude(data)

    print("\n|v| summary stats:")
    summary("train (0-6999)", train_field)
    summary("test (locked)", test_field)
    summary("full (0-7999)", full_field)

    # 1-Wasserstein between train and test |v| distributions (on capped
    # subset for speed). If this is small, the test set is representative
    # and the eval WD blowup is NOT a test-set issue.
    cap = 100_000
    train_flat = train_field.flatten()
    test_flat = test_field.flatten()
    rng = np.random.RandomState(0)
    train_sub = rng.choice(train_flat, size=min(cap, len(train_flat)), replace=False)
    test_sub = rng.choice(test_flat, size=min(cap, len(test_flat)), replace=False)
    wd = float(wasserstein_distance(train_sub, test_sub))
    print(f"\n1-Wasserstein(train |v|, test |v|) = {wd:.4f}")
    print("  (the eval reported WD ~0.49 across methods; if this number is")
    print("   on that order, the test set is the problem.)")

    # Side-by-side histogram.
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, max(full_field.max(), 1.5), 100)
    ax.hist(train_flat, bins=bins, density=True, alpha=0.4,
            label=f"train ({train_n} frames)", color="steelblue")
    ax.hist(test_flat, bins=bins, density=True, histtype="step", linewidth=2,
            label=f"test ({len(test_idx)} frames)", color="crimson")
    ax.set_xlabel("|v|")
    ax.set_ylabel("Density")
    ax.set_title(f"NS |v| distribution: train vs locked test set "
                 f"(WD={wd:.4f})")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "v_hist.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
