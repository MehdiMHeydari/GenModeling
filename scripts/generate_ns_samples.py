"""
Generate a single NS sample comparison figure for the paper.

One GT row at top, one Gen row per (method, step_count) below — same noise
seed across all rows so columns align. Skips the teacher (75-step sampling
is slow) and only samples a curated set of paper-worthy methods. Edit
DEFAULT_METHODS below or pass --only to override.

Usage:
    python scripts/generate_ns_samples.py --gpu 6
    python scripts/generate_ns_samples.py --gpu 6 \\
        --only "NS-Reflow-ckpt200@1,NS-RF-ckpt799@10"
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

from src.eval.constants import get_dataset, load_test_indices
from scripts.evaluate_paper import (
    KIND_SAMPLERS, denormalize, make_noise, make_unet_cfg, resolve_ckpt,
    to_scalar_field,
)


N_SHOW = 8
SEED = 0

# Curated list of (method_name, step_count) to include. Picked from the
# 2026-04-27 eval — best of each family, no teacher.
DEFAULT_METHODS = [
    ("NS-Reflow-ckpt200", 1),    # winner: WD 0.0102 at 1 NFE
    ("NS-Reflow-ckpt200", 2),    # also strong
    ("NS-RF-ckpt799", 10),       # best RF round 1 (WD 0.016)
    ("NS-MFM-ckpt725", 16),      # latest MFM (WD 0.034)
    ("NS-CD16-ckpt999", 16),     # baseline
    ("NS-MM21-ckpt75", 16),      # MM diversity-lift winner
    ("NS-MM22-ckpt75", 16),      # other MM
]


def parse_only(s):
    out = []
    for token in s.split(","):
        token = token.strip()
        if "@" not in token:
            raise SystemExit(f"--only entries must be 'method@steps', got '{token}'")
        name, steps = token.split("@", 1)
        out.append((name.strip(), int(steps.strip())))
    return out


def plot_combined(gt_mag, gen_rows, path):
    """gen_rows: list of (label, gen_mag_array). One GT row + one Gen row per method."""
    n_methods = len(gen_rows)
    n_rows = 1 + n_methods
    fig, axes = plt.subplots(n_rows, N_SHOW,
                             figsize=(2.2 * N_SHOW, 2.4 * n_rows),
                             squeeze=False)

    for j in range(N_SHOW):
        # Per-column shared vmax across GT and all Gen rows so each column
        # is internally comparable; columns can have different vmax because
        # NS |v| is heavy-tailed (one frame can have 3x the energy of others).
        col_vals = [gt_mag[j].max()] + [gen[j].max() for _, gen in gen_rows]
        vmax = max(col_vals)

        axes[0, j].imshow(gt_mag[j], vmin=0.0, vmax=vmax)
        axes[0, j].axis("off")
        for i, (_, gen_mag) in enumerate(gen_rows, start=1):
            axes[i, j].imshow(gen_mag[j], vmin=0.0, vmax=vmax)
            axes[i, j].axis("off")

    axes[0, 0].set_ylabel("GT |v|", fontsize=10, rotation=0,
                          labelpad=80, ha="right", va="center")
    for i, (label, _) in enumerate(gen_rows, start=1):
        axes[i, 0].set_ylabel(label, fontsize=9, rotation=0,
                              labelpad=80, ha="right", va="center")
    # imshow + axis off hides ylabels — re-show them
    for i in range(n_rows):
        axes[i, 0].axis("on")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        for spine in axes[i, 0].spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--config", type=str, default="config/ns_paper_eval.yaml")
    p.add_argument("--output", type=str, default="paper_figures/ns_samples_combined.png")
    p.add_argument("--only", type=str, default=None,
                   help="Comma list of 'method@steps' tuples to override DEFAULT_METHODS")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if th.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    by_name = {m["name"]: m for m in cfg["methods"]}

    selection = parse_only(args.only) if args.only else DEFAULT_METHODS
    plan = []
    for name, steps in selection:
        if name not in by_name:
            print(f"[skip] not in config: {name}")
            continue
        plan.append((name, steps, by_name[name]))
    if not plan:
        raise SystemExit("Nothing to sample")

    dataset = cfg.get("dataset", "ns")
    ds = get_dataset(dataset)
    data_shape = ds["data_shape"]
    unet_cfg = make_unet_cfg(data_shape)

    data_min = np.load(os.path.join(ds["stats_dir"], "data_min.npy"))
    data_max = np.load(os.path.join(ds["stats_dir"], "data_max.npy"))
    test_idx = load_test_indices(dataset)[:N_SHOW]
    with h5py.File(ds["data_path"], "r") as f:
        data = f["tensor"][:]
    if data.ndim == 3:
        data = data[:, None]
    real_denorm = data[test_idx]
    gt_mag = to_scalar_field(real_denorm)

    noise = make_noise(SEED, N_SHOW, data_shape)

    gen_rows = []
    for name, steps, entry in plan:
        kind = entry["kind"]
        sampler = KIND_SAMPLERS[kind]
        ckpt = resolve_ckpt(entry)
        if kind == "cd":
            samples, nfe = sampler(ckpt, entry["student_steps"], noise,
                                   device, unet_cfg, batch_size=N_SHOW)
        else:
            samples, nfe = sampler(ckpt, steps, noise, device, unet_cfg,
                                   batch_size=N_SHOW)
        gen_denorm = denormalize(samples, data_min, data_max)
        gen_mag = to_scalar_field(gen_denorm)
        label = f"{name}\n({steps}s, NFE={nfe})"
        gen_rows.append((label, gen_mag))
        print(f"  sampled {name} @ {steps} steps")

    plot_combined(gt_mag, gen_rows, args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
