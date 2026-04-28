"""
Generate per-method sample grids for the NS paper.

For each method in `config/ns_paper_eval.yaml`, draws 8 samples with seed 0,
denormalizes, and saves a 2-row velocity-magnitude grid (GT on top, Gen on
bottom) so the eval numbers can be visually sanity-checked. Reuses the
sampler dispatch from `scripts/evaluate_paper.py` so model-loading stays in
one place.

For methods with multiple `step_counts`, generates one grid per step count.

Usage:
    python scripts/generate_ns_samples.py --gpu 2
    python scripts/generate_ns_samples.py --gpu 2 \\
        --only "NS-Reflow-ckpt200,NS-Teacher-ckpt75"
    python scripts/generate_ns_samples.py --gpu 2 \\
        --output_dir paper_figures/ns_samples_v1
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

from src.eval.constants import SCHEDULE_S, get_dataset, load_test_indices
from scripts.evaluate_paper import (
    KIND_SAMPLERS, denormalize, make_noise, make_unet_cfg, resolve_ckpt,
    to_scalar_field,
)


N_SHOW = 8
SEED = 0


def gt_grid(real_denorm, n_show):
    return to_scalar_field(real_denorm[:n_show])


def plot_method(gt_mag, gen_mag, path, title):
    """Each pair (GT_j, Gen_j) shares a vmax so structure within a pair is
    comparable, but pairs do NOT share vmax — NS |v| is heavy-tailed (max
    ~3.6 vs p99 ~0.9), so a single global vmax washes calmer frames into a
    near-flat dim color."""
    fig, axes = plt.subplots(2, gen_mag.shape[0], figsize=(2.2 * gen_mag.shape[0], 5.0))
    for j in range(gen_mag.shape[0]):
        vmax = max(gt_mag[j].max(), gen_mag[j].max())
        axes[0, j].imshow(gt_mag[j], vmin=0.0, vmax=vmax)
        axes[0, j].set_title("GT |v|", fontsize=8)
        axes[0, j].axis("off")
        axes[1, j].imshow(gen_mag[j], vmin=0.0, vmax=vmax)
        axes[1, j].set_title("Gen |v|", fontsize=8)
        axes[1, j].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--config", type=str, default="config/ns_paper_eval.yaml")
    p.add_argument("--output_dir", type=str, default="paper_figures/ns_samples")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated method names (subset of config)")
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
        if not methods:
            raise SystemExit(f"No methods matched --only={args.only}")

    dataset = cfg.get("dataset", "ns")
    ds = get_dataset(dataset)
    data_shape = ds["data_shape"]
    unet_cfg = make_unet_cfg(data_shape)

    # Load just enough GT to populate the grid.
    data_min = np.load(os.path.join(ds["stats_dir"], "data_min.npy"))
    data_max = np.load(os.path.join(ds["stats_dir"], "data_max.npy"))
    test_idx = load_test_indices(dataset)[:N_SHOW]
    with h5py.File(ds["data_path"], "r") as f:
        data = f["tensor"][:]
    if data.ndim == 3:
        data = data[:, None]
    real_denorm = data[test_idx]
    gt_mag = gt_grid(real_denorm, N_SHOW)

    noise = make_noise(SEED, N_SHOW, data_shape)

    for entry in methods:
        kind = entry["kind"]
        if kind not in KIND_SAMPLERS:
            print(f"[skip] unknown kind: {kind} ({entry['name']})")
            continue
        try:
            ckpt = resolve_ckpt(entry)
        except FileNotFoundError as e:
            print(f"[skip] {entry['name']}: {e}")
            continue

        sampler = KIND_SAMPLERS[kind]
        step_counts = entry.get("step_counts", [entry.get("student_steps")])

        for n_steps in step_counts:
            if kind == "cd":
                samples, nfe = sampler(
                    ckpt, entry["student_steps"], noise, device, unet_cfg,
                    batch_size=N_SHOW,
                )
            else:
                samples, nfe = sampler(ckpt, n_steps, noise, device, unet_cfg,
                                       batch_size=N_SHOW)
            gen_denorm = denormalize(samples, data_min, data_max)
            gen_mag = to_scalar_field(gen_denorm)

            tag = f"{entry['name']}_steps{n_steps}"
            title = f"{entry['name']}  |  {n_steps} steps  |  NFE={nfe}"
            out = os.path.join(args.output_dir, f"{tag}.png")
            plot_method(gt_mag, gen_mag, out, title)
            print(f"  wrote {out}")

    print(f"\nDone. Sample grids in {args.output_dir}/")


if __name__ == "__main__":
    main()
