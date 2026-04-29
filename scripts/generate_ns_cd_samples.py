"""
NS CD-focused sample figure: Teacher + CD-4 (latest) + CD-16 (final).

Standalone variant of generate_ns_samples.py that only includes the
teacher and the available CD students. CD-8 not included because we
never trained it on NS.

CD-4 is currently being trained on GPU 1 — script picks the latest
checkpoint at run-time so this can be re-run as training progresses.
CD-16 uses the final checkpoint (epoch 999) from exp_1.

Usage:
    python scripts/generate_ns_cd_samples.py --gpu 6
"""

import argparse
import os

import h5py
import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.constants import get_dataset, load_test_indices
from scripts.evaluate_paper import (
    KIND_SAMPLERS, denormalize, latest_checkpoint, make_noise, make_unet_cfg,
    to_scalar_field,
)


N_SHOW = 8
SEED = 0

# Hardcoded method specs.
# Each entry: (display_name, kind, ckpt_or_None, exp_dir_or_None,
#              student_steps, step_count)
METHODS = [
    ("Teacher", "teacher",
     "ns_teacher/exp_1/saved_state/checkpoint_75.pt", None, None, 75),
    # CD-4 still training (exp_2). Auto-pick latest checkpoint each run.
    ("CD-4", "cd", None, "ns_student/exp_2", 4, 4),
    # CD-16 fully trained (exp_1, epoch 999).
    ("CD-16", "cd",
     "ns_student/exp_1/saved_state/checkpoint_999.pt", None, 16, 16),
]


def plot_combined(gt_mag, gen_rows, path):
    n_methods = len(gen_rows)
    n_rows = 1 + n_methods
    fig, axes = plt.subplots(n_rows, N_SHOW,
                             figsize=(2.2 * N_SHOW, 2.4 * n_rows),
                             squeeze=False)

    for j in range(N_SHOW):
        axes[0, j].imshow(gt_mag[j])
        axes[0, j].axis("off")
        for i, (_, gen_mag) in enumerate(gen_rows, start=1):
            axes[i, j].imshow(gen_mag[j])
            axes[i, j].axis("off")

    axes[0, 0].set_ylabel("GT |v|", fontsize=10, rotation=0,
                          labelpad=80, ha="right", va="center")
    for i, (label, _) in enumerate(gen_rows, start=1):
        axes[i, 0].set_ylabel(label, fontsize=9, rotation=0,
                              labelpad=80, ha="right", va="center")
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
    p.add_argument("--output", type=str,
                   default="paper_figures/ns_cd_samples.png")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if th.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ds = get_dataset("ns")
    data_shape = ds["data_shape"]
    unet_cfg = make_unet_cfg(data_shape)

    data_min = np.load(os.path.join(ds["stats_dir"], "data_min.npy"))
    data_max = np.load(os.path.join(ds["stats_dir"], "data_max.npy"))
    full_test_idx = load_test_indices("ns")
    spread = np.linspace(0, len(full_test_idx) - 1, N_SHOW).astype(int)
    test_idx = full_test_idx[spread]
    with h5py.File(ds["data_path"], "r") as f:
        data = f["tensor"][:]
    if data.ndim == 3:
        data = data[:, None]
    real_denorm = data[test_idx]
    gt_mag = to_scalar_field(real_denorm)

    noise = make_noise(SEED, N_SHOW, data_shape)

    gen_rows = []
    for name, kind, ckpt_path, exp_dir, student_steps, n_steps in METHODS:
        if ckpt_path is None:
            ckpt_path = latest_checkpoint(os.path.join(exp_dir, "saved_state"))
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"No checkpoint in {exp_dir}/saved_state — "
                    f"is {name} still training?")
        sampler = KIND_SAMPLERS[kind]
        if kind == "cd":
            samples, nfe = sampler(ckpt_path, student_steps, noise, device,
                                   unet_cfg, batch_size=N_SHOW)
        else:
            samples, nfe = sampler(ckpt_path, n_steps, noise, device, unet_cfg,
                                   batch_size=N_SHOW)
        gen_denorm = denormalize(samples, data_min, data_max)
        gen_mag = to_scalar_field(gen_denorm)
        label = f"{name}\n({n_steps}s, NFE={nfe})"
        gen_rows.append((label, gen_mag))
        print(f"  sampled {name} ({os.path.basename(ckpt_path)})")

    plot_combined(gt_mag, gen_rows, args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
