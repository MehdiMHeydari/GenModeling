"""
CNS teacher checkpoint diagnostic sweep (mirrors the NS protocol).

Training is non-monotone (best epoch loss 0.027 vs final 0.048), and on NS
the canonical checkpoint turned out to be an EARLY one (ckpt 75, not the
latest). Before launching CD/PRISM we sweep checkpoints, sample each at
DDIM 75 from a fixed noise seed, and compare against the locked test set
in physical units:

  - WD: 1-Wasserstein on pixel marginals (all channels pooled), lower better
  - per-channel std ratio gen/real (dispersion; 1.0 = ideal, <1 collapsed)
  - positivity violation fraction (real ref: 0.0)
  - mean abs divergence (contrastive; real ref 0.029)

Outputs under diagnostics/cns_teacher_v1/:
  sweep_stats.csv / sweep_stats.txt   one row per checkpoint
  samples_ckpt<E>.png                 4x4 grid per checkpoint (vs real row)
  summary.png                         WD + std ratios vs epoch

Usage (server):
    PYTHONPATH=. python scripts/diagnose_cns_teacher.py --gpu 4
"""

import argparse
import csv
import os

import h5py
import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

OUT = "diagnostics/cns_teacher_v1"
CKPT_DIR = "cns_teacher/exp_1/saved_state"
DATA = "data/cns_128_merged.h5"
TEST_IDX = "data/cns_test_indices.npy"
CHANNELS = ["density", "pressure", "Vx", "Vy"]
EPOCHS = [75, 150, 300, 450, 599]
N_GEN = 256
DDIM_STEPS = 75
SEED = 0
WD_SUBSAMPLE = 200_000  # pixels per side for the WD estimate
REAL_DIV_REF = 0.029


def mean_abs_divergence(samples):
    vx, vy = samples[:, 2], samples[:, 3]
    dvx = (np.roll(vx, -1, axis=2) - np.roll(vx, 1, axis=2)) / 2.0
    dvy = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / 2.0
    return float(np.mean(np.abs(dvx + dvy)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if th.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    th.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    from scripts.evaluate_paper import sample_teacher, make_unet_cfg

    ch_mean = np.load(os.path.join(CKPT_DIR, "data_mean.npy")).astype(np.float32)
    ch_std = np.load(os.path.join(CKPT_DIR, "data_std.npy")).astype(np.float32)

    test_idx = np.sort(np.load(TEST_IDX))
    with h5py.File(DATA, "r") as f:
        real = f["tensor"][test_idx]  # (1000, 4, H, W) physical units
    real_flat = real.reshape(-1)
    real_sub = rng.choice(real_flat, WD_SUBSAMPLE, replace=False)
    real_std = real.std(axis=(0, 2, 3))

    unet_cfg = make_unet_cfg((4, 128, 128))
    noise = th.randn(N_GEN, 4, 128, 128)  # same noise for every checkpoint

    rows = []
    for ep in EPOCHS:
        ckpt = os.path.join(CKPT_DIR, f"checkpoint_{ep}.pt")
        samples, _ = sample_teacher(ckpt, DDIM_STEPS, noise.clone(), device, unet_cfg)
        gen = samples.numpy() * ch_std.reshape(1, -1, 1, 1) \
            + ch_mean.reshape(1, -1, 1, 1)

        gen_sub = rng.choice(gen.reshape(-1), WD_SUBSAMPLE, replace=False)
        wd = float(wasserstein_distance(gen_sub, real_sub))
        std_ratio = (gen.std(axis=(0, 2, 3)) / real_std)
        pos = float(((gen[:, 0] <= 0) | (gen[:, 1] <= 0)).mean())
        div = mean_abs_divergence(gen)

        row = {"epoch": ep, "wd": wd, "positivity_frac": pos, "div": div}
        for c, name in enumerate(CHANNELS):
            row[f"stdratio_{name}"] = float(std_ratio[c])
        rows.append(row)
        print(f"ckpt {ep:4d}: WD={wd:.4f}  pos_viol={pos:.5f}  div={div:.4f}  "
              f"std_ratio=" + " ".join(f"{name}:{std_ratio[c]:.2f}"
                                       for c, name in enumerate(CHANNELS)))

        # 4 real + 4 gen grid, per-channel scales
        fig, axes = plt.subplots(4, 8, figsize=(16.4, 8.6))
        for c in range(4):
            vmin, vmax = np.percentile(real[:, c], [1, 99])
            for j in range(4):
                axes[c, j].imshow(real[j, c], vmin=vmin, vmax=vmax, cmap="RdBu_r")
                axes[c, j].set_xticks([]); axes[c, j].set_yticks([])
                if c == 0: axes[c, j].set_title(f"real {j}", fontsize=9)
                if j == 0: axes[c, j].set_ylabel(CHANNELS[c], fontsize=10)
            for j in range(4):
                axes[c, 4 + j].imshow(gen[j, c], vmin=vmin, vmax=vmax, cmap="RdBu_r")
                axes[c, 4 + j].set_xticks([]); axes[c, 4 + j].set_yticks([])
                if c == 0: axes[c, 4 + j].set_title(f"ckpt{ep} {j}", fontsize=9)
        plt.suptitle(f"CNS teacher ckpt_{ep} @DDIM {DDIM_STEPS} vs real "
                     f"(WD {wd:.4f})", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{OUT}/samples_ckpt{ep}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # persist stats
    with open(f"{OUT}/sweep_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(f"{OUT}/sweep_stats.txt", "w") as fh:
        for r in rows:
            fh.write(str(r) + "\n")

    # summary figure
    eps = [r["epoch"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    ax1.plot(eps, [r["wd"] for r in rows], "o-", color="#4477AA")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("WD (physical units)")
    ax1.set_title("distribution match vs checkpoint (lower better)")
    for c, name in enumerate(CHANNELS):
        ax2.plot(eps, [r[f"stdratio_{name}"] for r in rows], "o-", label=name)
    ax2.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("std(gen)/std(real)")
    ax2.set_title("per-channel dispersion (1.0 ideal)")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/summary.png", dpi=130, bbox_inches="tight")
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
