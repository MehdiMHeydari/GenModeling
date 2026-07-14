"""
Quick CNS teacher check at the current (mid-training) checkpoint.

Samples 8 fields @ DDIM 75 from checkpoint_175, compares per-channel stats
and a visual grid against real test samples. Decision input for the
normalization question: if the teacher already reproduces pressure tails
and velocity amplitudes, min-max normalization is survivable; if it
under-disperses the compressed channels the same way MFM does, the
normalization is hurting every method and we retrain with a better scheme.

Usage (server):
    PYTHONPATH=. python scripts/diagnose_cns_teacher_quick.py --gpu 7
"""

import argparse
import os

import h5py
import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "diagnostics/cns_mfm_v1"
CKPT = "cns_teacher/exp_1/saved_state/checkpoint_175.pt"
DATA = "data/cns_128_merged.h5"
TEST_IDX = "data/cns_test_indices.npy"
STATS_DIR = "cns_teacher/exp_1/saved_state"
CHANNELS = ["density", "pressure", "Vx", "Vy"]
N_GEN = 8
N_SHOW = 4
DDIM_STEPS = 75
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=7)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if th.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    th.manual_seed(SEED)

    from scripts.evaluate_paper import sample_teacher, make_unet_cfg

    # z-score stats (per-channel), written by get_cns_loader
    ch_mean = np.load(os.path.join(STATS_DIR, "data_mean.npy")).astype(np.float32)
    ch_std = np.load(os.path.join(STATS_DIR, "data_std.npy")).astype(np.float32)

    unet_cfg = make_unet_cfg((4, 128, 128))
    noise = th.randn(N_GEN, 4, 128, 128)
    samples, _ = sample_teacher(CKPT, DDIM_STEPS, noise, device, unet_cfg)
    samples = samples.numpy()

    gen = samples * ch_std.reshape(1, -1, 1, 1) + ch_mean.reshape(1, -1, 1, 1)

    test_idx = np.load(TEST_IDX)[:N_GEN]
    with h5py.File(DATA, "r") as f:
        real = f["tensor"][np.sort(test_idx)]

    lines = []
    for tag, arr in [("real", real), ("teacher", gen)]:
        for c, name in enumerate(CHANNELS):
            x = arr[:, c]
            lines.append(
                f"{tag:8s} {name:8s} min={x.min():+9.3f} med={np.median(x):+9.3f} "
                f"mean={x.mean():+9.3f} max={x.max():+9.3f} std={x.std():8.3f}"
            )
        rho, p = arr[:, 0], arr[:, 1]
        lines.append(f"{tag:8s} positivity-violation frac = "
                     f"{float(((rho <= 0) | (p <= 0)).mean()):.6f}")
        lines.append("")
    stats = "\n".join(lines)
    print(stats)
    with open(f"{OUT}/teacher_stats.txt", "w") as fh:
        fh.write(stats)

    fig, axes = plt.subplots(4, 2 * N_SHOW, figsize=(2.1 * 2 * N_SHOW, 8.6))
    for c in range(4):
        vmin, vmax = np.percentile(real[:, c], [1, 99])
        for j in range(N_SHOW):
            ax = axes[c, j]
            ax.imshow(real[j, c], vmin=vmin, vmax=vmax, cmap="RdBu_r")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_title(f"real {j}", fontsize=9)
            if j == 0:
                ax.set_ylabel(CHANNELS[c], fontsize=10)
        for j in range(N_SHOW):
            ax = axes[c, N_SHOW + j]
            ax.imshow(gen[j, c], vmin=vmin, vmax=vmax, cmap="RdBu_r")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_title(f"teacher {j}", fontsize=9)
    plt.suptitle(f"CNS: real (left 4) vs teacher ckpt_175 @DDIM {DDIM_STEPS} "
                 "(right 4), shared per-channel color scale", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/teacher_vs_real.png", dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}/teacher_vs_real.png")


if __name__ == "__main__":
    main()
