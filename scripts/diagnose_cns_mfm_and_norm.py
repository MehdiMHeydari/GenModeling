"""
One-off diagnostic (CNS):
  A. Sample the MFM checkpoint_25 (1-step and 16-step) and compare fields
     against real test samples -- did MFM actually learn despite the
     adaptive loss sitting at ~1.0?
  B. Normalization analysis -- what does global-per-channel min-max do to
     the heavy-tailed pressure/density channels? Histograms in normalized
     space + a fixed-scale visual + effective-range stats.

Outputs (all under diagnostics/cns_mfm_v1/):
  mfm_vs_real.png        channel rows x (4 real | 4 gen 16-step) grid
  mfm_stats.txt          per-channel stats table + positivity check
  norm_histograms.png    per-channel histograms of normalized values
  norm_visual.png        a real sample in normalized space, fixed [-1,1] scale
  norm_stats.txt         effective-range table

Usage (server):
    python scripts/diagnose_cns_mfm_and_norm.py --gpu 7
"""

import argparse
import os

import h5py
import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.flow_models import MeanFlowMatching
from src.inference.samplers import MeanSampler

OUT = "diagnostics/cns_mfm_v1"
CKPT = "cns_mean_flow/exp_1/saved_state/checkpoint_25.pt"
DATA = "data/cns_128_merged.h5"
TEST_IDX = "data/cns_test_indices.npy"
STATS_DIR = "cns_mean_flow/exp_1/saved_state"  # per-channel min/max
CHANNELS = ["density", "pressure", "Vx", "Vy"]
N_SHOW = 4
N_GEN = 8
SEED = 0


def build_model(device):
    network = UNetModel(
        dim=[4, 128, 128],
        channel_mult="1,  2,  4,  4",
        num_channels=64,
        num_res_blocks=2,
        num_head_channels=32,
        attention_resolutions="32",
        dropout=0.0,
        use_new_attention_order=True,
        use_scale_shift_norm=True,
        class_cond=False,
        num_classes=None,
        use_future_time_emb=True,
    )
    model = MeanFlowMatching(network=network, infer=True)
    state = th.load(CKPT, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def denorm(x, ch_min, ch_max):
    # x: (N, C, H, W) in [-1, 1]; per-channel stats
    span = (ch_max - ch_min).reshape(1, -1, 1, 1)
    return (x + 1.0) / 2.0 * span + ch_min.reshape(1, -1, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=7)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if th.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    th.manual_seed(SEED)
    np.random.seed(SEED)

    ch_min = np.load(os.path.join(STATS_DIR, "data_min.npy")).astype(np.float32)
    ch_max = np.load(os.path.join(STATS_DIR, "data_max.npy")).astype(np.float32)
    print(f"per-channel min: {ch_min.tolist()}")
    print(f"per-channel max: {ch_max.tolist()}")

    # ---------- real samples ----------
    test_idx = np.load(TEST_IDX)[:N_GEN]
    with h5py.File(DATA, "r") as f:
        real = f["tensor"][np.sort(test_idx)]  # (N, 4, H, W) physical units

    # ---------- A. sample MFM ----------
    model = build_model(device)
    sampler = MeanSampler(model)
    gens = {}
    for steps in (1, 16):
        z = th.randn(N_GEN, 4, 128, 128, device=device)
        with th.no_grad():
            x = sampler.sample(z, t_span_kwargs={"start": 0, "end": 1,
                                                 "steps": steps + 1})
        gens[steps] = denorm(x.cpu().numpy(), ch_min, ch_max)
        print(f"sampled {N_GEN} @ {steps} step(s)")

    gen16 = gens[16]

    # grid: 4 channel rows x (N_SHOW real + N_SHOW gen)
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
            ax.imshow(gen16[j, c], vmin=vmin, vmax=vmax, cmap="RdBu_r")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_title(f"MFM-16 {j}", fontsize=9)
    plt.suptitle("CNS: real (left 4) vs MFM ckpt_25 @16 steps (right 4), "
                 "shared per-channel color scale", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/mfm_vs_real.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # stats table
    lines = []
    for tag, arr in [("real", real), ("gen16", gen16), ("gen1", gens[1])]:
        for c, name in enumerate(CHANNELS):
            x = arr[:, c]
            lines.append(
                f"{tag:6s} {name:8s} min={x.min():+9.3f} med={np.median(x):+9.3f} "
                f"mean={x.mean():+9.3f} max={x.max():+9.3f} std={x.std():8.3f}"
            )
        rho, p = arr[:, 0], arr[:, 1]
        frac = float(((rho <= 0) | (p <= 0)).mean())
        lines.append(f"{tag:6s} positivity-violation frac = {frac:.6f}")
        lines.append("")
    stats = "\n".join(lines)
    print(stats)
    with open(f"{OUT}/mfm_stats.txt", "w") as fh:
        fh.write(stats)

    # ---------- B. normalization analysis ----------
    with h5py.File(DATA, "r") as f:
        sub_idx = np.linspace(0, f["tensor"].shape[0] - 1, 500).astype(int)
        sub = f["tensor"][sub_idx]  # physical units
    span = (ch_max - ch_min).reshape(1, -1, 1, 1)
    sub_norm = 2.0 * (sub - ch_min.reshape(1, -1, 1, 1)) / span - 1.0

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    for c, name in enumerate(CHANNELS):
        axes[c].hist(sub_norm[:, c].ravel(), bins=200, log=True, color="#4477AA")
        axes[c].set_title(f"{name} (normalized)")
        axes[c].set_xlim(-1.05, 1.05)
    plt.suptitle("Histogram of normalized values the model actually sees "
                 "(log y, 500 frames)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/norm_histograms.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for c, name in enumerate(CHANNELS):
        im = axes[c].imshow(sub_norm[0, c], vmin=-1, vmax=1, cmap="RdBu_r")
        axes[c].set_title(f"{name}")
        axes[c].set_xticks([]); axes[c].set_yticks([])
    fig.colorbar(im, ax=axes.tolist(), shrink=0.85)
    plt.suptitle("One real sample in NORMALIZED space, fixed [-1,1] scale "
                 "(low contrast = channel compressed)", fontsize=11)
    plt.savefig(f"{OUT}/norm_visual.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    lines = []
    for c, name in enumerate(CHANNELS):
        x = sub_norm[:, c].ravel()
        p1, p50, p99 = np.percentile(x, [1, 50, 99])
        lines.append(
            f"{name:8s} std={x.std():.4f}  median={p50:+.4f}  "
            f"p1..p99 span={p99 - p1:.4f} of 2.0  "
            f"({100 * (p99 - p1) / 2:.1f}% of range)"
        )
    norm_stats = "\n".join(lines)
    print(norm_stats)
    with open(f"{OUT}/norm_stats.txt", "w") as fh:
        fh.write(norm_stats)

    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
