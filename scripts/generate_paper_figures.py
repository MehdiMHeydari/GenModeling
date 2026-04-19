"""
Visual sample comparison figures for the paper.

Samples each method (from config/paper_eval.yaml) using the locked test
indices + seed from src.eval.constants, then produces:

  results/figures/sample_grid.png   — one row per method, N_SHOW columns
  results/figures/histograms.png    — marginal distribution per method

This uses the same sampling code as evaluate_paper.py so figures stay
consistent with the numeric results.

Usage:
    # Default: seed 0, 8 samples shown per method, canonical step counts
    python scripts/generate_paper_figures.py --gpu 0

    # Subset of methods, different number of samples shown
    python scripts/generate_paper_figures.py --gpu 0 \\
        --only "Teacher,MM-exp22" --n_show 12
"""

import argparse
import os

import h5py
import numpy as np
import torch as th
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.constants import (
    DATA_PATH, DATA_SHAPE, N_TEST_SAMPLES, SCHEDULE_S, STATS_DIR,
    SINGLE_SEED, load_test_indices,
)
from scripts.evaluate_paper import (
    KIND_SAMPLERS, denormalize, make_noise, resolve_ckpt,
)

CMAP = "RdBu_r"

# For each method, the single step count to use for the visual figure.
# Picked to reflect the method's paper-story setting (e.g. Teacher=250,
# RF=few-step). Override per-entry in the config if needed.
CANONICAL_STEPS = {
    "Teacher": 250,
    "CD-4step": 4,
    "CD-8step": 8,
    "CD-16step": 16,
    "MM-exp20": 16,
    "MM-exp21": 16,
    "MM-exp22": 16,
    "RF": 5,
    "Reflow": 5,
    "MFM": 16,
}


def sample_one(entry, n_samples, seed, device):
    kind = entry["kind"]
    sampler = KIND_SAMPLERS[kind]
    ckpt = resolve_ckpt(entry)
    noise = make_noise(seed, n_samples, DATA_SHAPE)

    if kind == "cd":
        samples, nfe = sampler(ckpt, entry["student_steps"], noise, device)
    else:
        n_steps = CANONICAL_STEPS.get(entry["name"], entry["step_counts"][-1])
        samples, nfe = sampler(ckpt, n_steps, noise, device)
    return samples, nfe


def plot_sample_grid(gt, method_samples, n_show, out_path):
    """
    Row 0 = ground truth, one row per method.
    Per-tile autoscale so each sample shows its own full dynamic range
    (matches the presentation-figure convention the user confirmed).
    """
    n_methods = len(method_samples)
    fig, axes = plt.subplots(
        n_methods + 1, n_show,
        figsize=(1.6 * n_show, 1.6 * (n_methods + 1)),
    )
    if n_methods + 1 == 1:
        axes = axes[None, :]

    for j in range(n_show):
        axes[0, j].imshow(gt[j, 0], cmap=CMAP)
        axes[0, j].axis("off")
    axes[0, 0].set_ylabel("Ground Truth", rotation=0, ha="right",
                          va="center", fontsize=10)

    for i, (name, samples, nfe) in enumerate(method_samples):
        for j in range(n_show):
            axes[i + 1, j].imshow(samples[j, 0], cmap=CMAP)
            axes[i + 1, j].axis("off")
        label = f"{name}\n(NFE={nfe})"
        # y-label on the first column only
        axes[i + 1, 0].text(
            -0.1, 0.5, label,
            transform=axes[i + 1, 0].transAxes,
            ha="right", va="center", fontsize=10,
        )

    # restore top-row label the same way
    axes[0, 0].text(
        -0.1, 0.5, "Ground Truth",
        transform=axes[0, 0].transAxes,
        ha="right", va="center", fontsize=10, fontweight="bold",
    )
    axes[0, 0].set_ylabel("")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_histograms(gt_flat, method_samples_denorm, out_path):
    n_methods = len(method_samples_denorm)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.8 * n_cols, 3.6 * n_rows),
                             squeeze=False)

    for idx, (name, denorm, nfe) in enumerate(method_samples_denorm):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r][c]
        ax.hist(gt_flat, bins=80, density=True, alpha=0.35,
                color="gray", label="Ground Truth")
        ax.hist(denorm.flatten(), bins=80, density=True,
                histtype="step", linewidth=2, label="Generated")
        ax.set_title(f"{name}  (NFE={nfe})")
        ax.set_xlabel("u(x, y)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for idx in range(len(method_samples_denorm), n_rows * n_cols):
        r, c = idx // n_cols, idx % n_cols
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--config", type=str, default="config/paper_eval.yaml")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated method names to render")
    p.add_argument("--n_show", type=int, default=8,
                   help="Number of samples shown per row in the grid")
    p.add_argument("--seed", type=int, default=SINGLE_SEED)
    p.add_argument("--output_dir", type=str, default="results/figures")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if th.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    methods = cfg["methods"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        methods = [m for m in methods if m["name"] in wanted]

    # Load ground-truth test samples (same indices as evaluate_paper.py)
    data_min = np.load(os.path.join(STATS_DIR, "data_min.npy"))
    data_max = np.load(os.path.join(STATS_DIR, "data_max.npy"))
    test_idx = load_test_indices()[:N_TEST_SAMPLES]
    with h5py.File(DATA_PATH, "r") as f:
        data = f["tensor"][:]
    if data.ndim == 3:
        data = data[:, None]
    gt_denorm = data[test_idx]

    n_samples = max(args.n_show, N_TEST_SAMPLES)

    # Sample each method
    sample_results = []
    for entry in methods:
        if entry["kind"] not in KIND_SAMPLERS:
            print(f"[skip] unknown kind: {entry['kind']}")
            continue
        try:
            resolve_ckpt(entry)
        except FileNotFoundError as e:
            print(f"[skip] {entry['name']}: {e}")
            continue
        print(f"Sampling {entry['name']}...")
        samples, nfe = sample_one(entry, n_samples, args.seed, device)
        denorm = denormalize(samples, data_min, data_max)
        sample_results.append((entry["name"], denorm, nfe))

    # Grid: first n_show samples per method
    grid_rows = [(n, s[:args.n_show], nfe) for n, s, nfe in sample_results]
    plot_sample_grid(
        gt_denorm[:args.n_show], grid_rows, args.n_show,
        os.path.join(args.output_dir, "sample_grid.png"),
    )

    # Histograms: full N_TEST_SAMPLES for stability
    plot_histograms(
        gt_denorm.flatten(), sample_results,
        os.path.join(args.output_dir, "histograms.png"),
    )

    print(f"\nDone. Figures in {args.output_dir}/")


if __name__ == "__main__":
    main()
