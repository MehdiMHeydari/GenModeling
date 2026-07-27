"""
CNS visual gallery: 16 samples per method vs 16 ground-truth fields.

One figure per channel (density, pressure, Vx, Vy). Rows: real, Teacher
(DDIM 250), CD-16, RF-10, Reflow-1, MFM-16. Columns: 16 samples (fixed
noise seed shared across methods, so column j uses the same latent for
every method). Also saves the sampled tensors to gallery_samples.pt for
reuse.

Usage (server):
    PYTHONPATH=. python scripts/generate_cns_gallery.py --gpu 5
"""

import argparse
import os

import h5py
import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "diagnostics/cns_gallery_v1"
STATS_DIR = "cns_teacher/exp_1/saved_state"
DATA = "data/cns_128_merged.h5"
TEST_IDX = "data/cns_test_indices.npy"
CHANNELS = ["density", "pressure", "Vx", "Vy"]
N = 16
SEED = 0

MODELS = [
    # (label, kind, ckpt, steps)
    ("Teacher-250", "teacher", "cns_teacher/exp_1/saved_state/checkpoint_599.pt", 250),
    ("CD-16",       "cd",      "cns_student/exp_1/saved_state/checkpoint_999.pt", 16),
    ("RF-10",       "rf",      "cns_rectified_flow/exp_1/saved_state/checkpoint_799.pt", 10),
    ("Reflow-1",    "rf",      "cns_rectified_flow_reflow/exp_1/saved_state/checkpoint_399.pt", 1),
    ("MFM-16",      "mfm",     "cns_mean_flow/exp_1/saved_state/checkpoint_999.pt", 16),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"
    os.makedirs(OUT, exist_ok=True)

    from scripts.evaluate_paper import KIND_SAMPLERS, make_unet_cfg

    ch_mean = np.load(os.path.join(STATS_DIR, "data_mean.npy")).astype(np.float32)
    ch_std = np.load(os.path.join(STATS_DIR, "data_std.npy")).astype(np.float32)

    test_idx = np.sort(np.load(TEST_IDX))[:N]
    with h5py.File(DATA, "r") as f:
        real = f["tensor"][test_idx]  # (16, 4, H, W)

    unet_cfg = make_unet_cfg((4, 128, 128))
    th.manual_seed(SEED)
    noise = th.randn(N, 4, 128, 128)

    all_gen = {"real": real}
    for label, kind, ckpt, steps in MODELS:
        sampler = KIND_SAMPLERS[kind]
        s, nfe = sampler(ckpt, steps, noise.clone(), device, unet_cfg,
                         batch_size=16)
        gen = s.numpy() * ch_std.reshape(1, -1, 1, 1) \
            + ch_mean.reshape(1, -1, 1, 1)
        all_gen[label] = gen
        print(f"{label}: sampled {N} @ nfe={nfe}")

    th.save({k: th.from_numpy(np.ascontiguousarray(v)) for k, v in all_gen.items()},
            f"{OUT}/gallery_samples.pt")

    rows = ["real"] + [m[0] for m in MODELS]
    for c, ch_name in enumerate(CHANNELS):
        vmin, vmax = np.percentile(real[:, c], [1, 99])
        fig, axes = plt.subplots(len(rows), N,
                                 figsize=(N * 1.06, len(rows) * 1.18))
        for r, rname in enumerate(rows):
            arr = all_gen[rname]
            for j in range(N):
                ax = axes[r, j]
                ax.imshow(arr[j, c], vmin=vmin, vmax=vmax, cmap="RdBu_r")
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(rname, fontsize=7)
                if r == 0:
                    ax.set_title(str(j), fontsize=6)
        plt.suptitle(f"CNS {ch_name}: 16 samples per method vs ground truth "
                     "(shared color scale, shared noise per column)",
                     fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{OUT}/gallery_{ch_name}.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {OUT}/gallery_{ch_name}.png")


if __name__ == "__main__":
    main()
