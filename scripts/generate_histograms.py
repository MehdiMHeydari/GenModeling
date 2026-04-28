"""
Per-method histograms for the paper.

For each (method, step_count) in the corresponding sample-grid script's
DEFAULT_METHODS, samples N_SAMPLES frames and plots histograms of the
relevant scalar field(s) overlaid against ground truth. Outputs both:
  - panel grid (one subplot per method)
  - single-axes overlay (all methods + GT on one plot)

Darcy (1-channel porosity) gets one panel + one overlay.
NS (2-channel velocity) gets three panel + three overlay images, one for
each of |v|, Vx, Vy.

Usage:
    python scripts/generate_histograms.py --gpu 6 --dataset ns
    python scripts/generate_histograms.py --gpu 6 --dataset darcy
"""

import argparse
import math
import os

import h5py
import numpy as np
import torch as th
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.constants import get_dataset, load_test_indices
from scripts.evaluate_paper import (
    KIND_SAMPLERS, denormalize, make_noise, make_unet_cfg, resolve_ckpt,
)


N_SAMPLES = 300
SEED = 0
N_BINS = 80


def get_default_methods(dataset):
    if dataset == "ns":
        from scripts.generate_ns_samples import DEFAULT_METHODS
    elif dataset == "darcy":
        from scripts.generate_darcy_samples import DEFAULT_METHODS
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DEFAULT_METHODS


def get_field_specs(dataset):
    """Return list of (field_name, field_label, x_axis_label) tuples."""
    if dataset == "ns":
        return [
            ("v", "|v|", "|v|"),
            ("vx", "Vx", "Vx"),
            ("vy", "Vy", "Vy"),
        ]
    elif dataset == "darcy":
        return [("porosity", "porosity", "porosity")]
    else:
        raise ValueError(dataset)


def extract_field(samples, field_name):
    """Pull a 1D scalar array out of (N, C, H, W) samples for the named field."""
    if field_name == "v":
        return np.sqrt(samples[:, 0] ** 2 + samples[:, 1] ** 2).flatten()
    elif field_name == "vx":
        return samples[:, 0].flatten()
    elif field_name == "vy":
        return samples[:, 1].flatten()
    elif field_name == "porosity":
        return samples[:, 0].flatten()
    raise ValueError(field_name)


def common_bins(*arrays, n_bins=N_BINS):
    lo = min(a.min() for a in arrays)
    hi = max(a.max() for a in arrays)
    return np.linspace(lo, hi, n_bins + 1)


def plot_panel(gt_vals, method_rows, x_label, path):
    """method_rows: list of (label, gen_vals)."""
    n = len(method_rows)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.0 * n_cols, 3.4 * n_rows),
                             squeeze=False)

    for idx, (label, gen_vals) in enumerate(method_rows):
        ax = axes[idx // n_cols, idx % n_cols]
        bins = common_bins(gt_vals, gen_vals)
        ax.hist(gt_vals, bins=bins, density=True, alpha=0.35,
                label="GT", color="gray")
        ax.hist(gen_vals, bins=bins, density=True, histtype="step",
                linewidth=2, label="Gen", color="C0")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Density")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)

    # blank out unused panels
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(gt_vals, method_rows, x_label, path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = common_bins(gt_vals, *[v for _, v in method_rows])
    ax.hist(gt_vals, bins=bins, density=True, alpha=0.35,
            label="GT", color="gray")
    cmap = plt.get_cmap("tab10")
    for i, (label, gen_vals) in enumerate(method_rows):
        ax.hist(gen_vals, bins=bins, density=True, histtype="step",
                linewidth=1.8, label=label, color=cmap(i % 10))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dataset", type=str, required=True,
                   choices=["ns", "darcy"])
    p.add_argument("--output_dir", type=str, default="paper_figures/histograms")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if th.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    config_path = ("config/ns_paper_eval.yaml" if args.dataset == "ns"
                   else "config/paper_eval.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    by_name = {m["name"]: m for m in cfg["methods"]}

    method_list = get_default_methods(args.dataset)
    plan = []
    for name, steps in method_list:
        if name not in by_name:
            print(f"[skip] not in config: {name}")
            continue
        plan.append((name, steps, by_name[name]))

    ds = get_dataset(args.dataset)
    data_shape = ds["data_shape"]
    unet_cfg = make_unet_cfg(data_shape)

    data_min = np.load(os.path.join(ds["stats_dir"], "data_min.npy"))
    data_max = np.load(os.path.join(ds["stats_dir"], "data_max.npy"))
    full_test_idx = load_test_indices(args.dataset)
    spread = np.linspace(0, len(full_test_idx) - 1, N_SAMPLES).astype(int)
    test_idx = full_test_idx[spread]
    with h5py.File(ds["data_path"], "r") as f:
        data = f["tensor"][:]
    if data.ndim == 3:
        data = data[:, None]
    real_denorm = data[test_idx]

    noise = make_noise(SEED, N_SAMPLES, data_shape)

    method_samples = []  # list of (label, gen_denorm)
    for name, steps, entry in plan:
        kind = entry["kind"]
        sampler = KIND_SAMPLERS[kind]
        ckpt = resolve_ckpt(entry)
        if kind == "cd":
            samples, nfe = sampler(ckpt, entry["student_steps"], noise,
                                   device, unet_cfg, batch_size=64)
        else:
            samples, nfe = sampler(ckpt, steps, noise, device, unet_cfg,
                                   batch_size=64)
        gen_denorm = denormalize(samples, data_min, data_max)
        label = f"{name} ({steps}s, NFE={nfe})"
        method_samples.append((label, gen_denorm))
        print(f"  sampled {name} @ {steps} steps  ({N_SAMPLES} frames)")

    for field_name, field_label, x_label in get_field_specs(args.dataset):
        gt_vals = extract_field(real_denorm, field_name)
        method_rows = [(label, extract_field(g, field_name))
                       for label, g in method_samples]

        suffix = f"_{field_name}" if args.dataset == "ns" else ""
        panel_path = os.path.join(
            args.output_dir, f"{args.dataset}_hist_panel{suffix}.png")
        overlay_path = os.path.join(
            args.output_dir, f"{args.dataset}_hist_overlay{suffix}.png")

        plot_panel(gt_vals, method_rows, x_label, panel_path)
        plot_overlay(gt_vals, method_rows, x_label, overlay_path)
        print(f"  wrote {panel_path}")
        print(f"  wrote {overlay_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
