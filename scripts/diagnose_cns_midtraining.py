"""
Mid-training sanity check for the CNS benchmark runs.

Samples every model that has a usable checkpoint right now and compares
against the locked test set in physical units. Cheap (1-16 NFE per sample,
192 samples per model) so it can share a GPU with a training job.

Models covered:
  RF ckpt_799 (COMPLETE)  @ 1 and 10 Euler steps
  CD exp_1  ckpt_250 (mid-training) @ 16 steps
  MFM       ckpt_375 (mid-training) @ 16 steps
  (teacher numbers from diagnostics/cns_teacher_v1 cited for context;
   not resampled here)

Metrics per model: WD vs real (200k-pixel subsample), per-channel std
ratio, positivity violation fraction, mean abs divergence
(real refs: WD baseline 0.40, positivity 0, divergence 0.029).

Outputs under diagnostics/cns_midtraining_v1/.

Usage (server):
    PYTHONPATH=. python scripts/diagnose_cns_midtraining.py --gpu 4
"""

import argparse
import os

import h5py
import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

OUT = "diagnostics/cns_midtraining_v1"
STATS_DIR = "cns_teacher/exp_1/saved_state"
DATA = "data/cns_128_merged.h5"
TEST_IDX = "data/cns_test_indices.npy"
CHANNELS = ["density", "pressure", "Vx", "Vy"]
N_GEN = 192
SEED = 0
WD_SUB = 200_000

MODELS = [
    # (label, kind, ckpt, steps)
    ("RF-1",     "rf",  "cns_rectified_flow/exp_1/saved_state/checkpoint_799.pt", 1),
    ("RF-10",    "rf",  "cns_rectified_flow/exp_1/saved_state/checkpoint_799.pt", 10),
    ("Reflow-1", "rf",  "cns_rectified_flow_reflow/exp_1/saved_state/checkpoint_399.pt", 1),
    ("Reflow-5", "rf",  "cns_rectified_flow_reflow/exp_1/saved_state/checkpoint_399.pt", 5),
    ("CD-16 final",  "cd",  "cns_student/exp_1/saved_state/checkpoint_999.pt", 16),
    ("MFM-16 final", "mfm", "cns_mean_flow/exp_1/saved_state/checkpoint_999.pt", 16),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if th.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    th.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    from scripts.evaluate_paper import KIND_SAMPLERS, make_unet_cfg

    ch_mean = np.load(os.path.join(STATS_DIR, "data_mean.npy")).astype(np.float32)
    ch_std = np.load(os.path.join(STATS_DIR, "data_std.npy")).astype(np.float32)

    test_idx = np.sort(np.load(TEST_IDX))
    with h5py.File(DATA, "r") as f:
        real = f["tensor"][test_idx]
    real_sub = rng.choice(real.reshape(-1), WD_SUB, replace=False)
    real_std = real.std(axis=(0, 2, 3))

    unet_cfg = make_unet_cfg((4, 128, 128))
    noise = th.randn(N_GEN, 4, 128, 128)

    lines = ["real refs: WD baseline 0.40 | positivity 0 | divergence 0.029",
             "teacher (ckpt599@250, from sweep): WD 2.41 | pos 1e-5 | "
             "std ratios 0.73/0.62/0.75/0.54", ""]
    print("\n".join(lines))

    for label, kind, ckpt, steps in MODELS:
        if not os.path.exists(ckpt):
            print(f"[skip] {label}: {ckpt} missing")
            continue
        sampler = KIND_SAMPLERS[kind]
        if kind == "cd":
            s, nfe = sampler(ckpt, steps, noise.clone(), device, unet_cfg,
                             batch_size=32)
        else:
            s, nfe = sampler(ckpt, steps, noise.clone(), device, unet_cfg,
                             batch_size=32)
        gen = s.numpy() * ch_std.reshape(1, -1, 1, 1) + ch_mean.reshape(1, -1, 1, 1)

        gsub = rng.choice(gen.reshape(-1), WD_SUB, replace=False)
        wd = float(wasserstein_distance(gsub, real_sub))
        sr = gen.std(axis=(0, 2, 3)) / real_std
        pos = float(((gen[:, 0] <= 0) | (gen[:, 1] <= 0)).mean())
        vx, vy = gen[:, 2], gen[:, 3]
        dvx = (np.roll(vx, -1, axis=2) - np.roll(vx, 1, axis=2)) / 2.0
        dvy = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / 2.0
        div = float(np.mean(np.abs(dvx + dvy)))

        line = (f"{label:16s} nfe={nfe:3d}  WD={wd:.4f}  pos={pos:.5f}  "
                f"div={div:.4f}  std_ratio="
                + " ".join(f"{n}:{sr[c]:.2f}" for c, n in enumerate(CHANNELS)))
        lines.append(line)
        print(line)

        fig, axes = plt.subplots(4, 8, figsize=(16.4, 8.6))
        for c in range(4):
            vmin, vmax = np.percentile(real[:, c], [1, 99])
            for j in range(4):
                axes[c, j].imshow(real[j, c], vmin=vmin, vmax=vmax, cmap="RdBu_r")
                axes[c, j].set_xticks([]); axes[c, j].set_yticks([])
                if c == 0: axes[c, j].set_title(f"real {j}", fontsize=9)
                if j == 0: axes[c, j].set_ylabel(CHANNELS[c], fontsize=10)
            for j in range(4):
                axes[c, 4 + j].imshow(gen[j, c], vmin=vmin, vmax=vmax,
                                      cmap="RdBu_r")
                axes[c, 4 + j].set_xticks([]); axes[c, 4 + j].set_yticks([])
                if c == 0: axes[c, 4 + j].set_title(f"{label} {j}", fontsize=9)
        plt.suptitle(f"CNS {label} vs real (WD {wd:.3f})", fontsize=11)
        plt.tight_layout()
        safe = label.replace(" ", "").replace("(", "_").replace(")", "")
        plt.savefig(f"{OUT}/{safe}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    with open(f"{OUT}/midtraining_stats.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
