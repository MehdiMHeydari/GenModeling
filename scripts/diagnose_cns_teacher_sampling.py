"""
CNS teacher sampling-protocol sweep: raw vs EMA weights x DDIM/DDPM.

The canonical teacher (ckpt 599 @ DDIM 250, raw weights) under-disperses
(std ratios 0.54-0.75). Two untested levers, both inference-time:
  - EMA weights (0.9999, saved in every checkpoint, never evaluated on CNS)
  - stochastic (ancestral / eta=1) sampling: deterministic DDIM is known to
    under-disperse; DDPM-style noise injection improves distributional
    coverage on tails
Also tests DDIM 500 for a step-count ceiling check.

Grid: {raw, ema} x {ddim250, ddim500, ddpm250}, 192 samples each, fixed
noise seed. Metrics vs the locked test set (real refs: WD baseline 0.40,
positivity 0, divergence 0.029).

Usage (server):
    PYTHONPATH=. python scripts/diagnose_cns_teacher_sampling.py --gpu 4
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

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.diffusion_utils import alpha_t, sigma_t, _broadcast_to_spatial

OUT = "diagnostics/cns_teacher_sampling_v1"
CKPT = "cns_teacher/exp_1/saved_state/checkpoint_599.pt"
STATS_DIR = "cns_teacher/exp_1/saved_state"
DATA = "data/cns_128_merged.h5"
TEST_IDX = "data/cns_test_indices.npy"
CHANNELS = ["density", "pressure", "Vx", "Vy"]
SCHEDULE_S = 0.008
N_GEN = 192
SEED = 0
WD_SUB = 200_000

CONFIGS = [
    ("raw",  "ddim", 250),
    ("ema",  "ddim", 250),
    ("raw",  "ddim", 500),
    ("ema",  "ddim", 500),
    ("raw",  "ddpm", 250),
    ("ema",  "ddpm", 250),
]


def make_unet():
    return UNetModel(
        dim=[4, 128, 128], channel_mult="1, 2, 4, 4", num_channels=64,
        num_res_blocks=2, num_head_channels=32, attention_resolutions="32",
        dropout=0.0, use_new_attention_order=True, use_scale_shift_norm=True,
        class_cond=False, num_classes=None,
    )


def generalized_step(x_hat, z_t, t, s, eta):
    """DDIM step with eta in [0,1]; eta=0 deterministic, eta=1 ancestral."""
    a_t = _broadcast_to_spatial(alpha_t(t, SCHEDULE_S), x_hat)
    sig_t = _broadcast_to_spatial(sigma_t(t, SCHEDULE_S), x_hat)
    a_s = _broadcast_to_spatial(alpha_t(s, SCHEDULE_S), x_hat)
    sig_s = _broadcast_to_spatial(sigma_t(s, SCHEDULE_S), x_hat)

    if eta == 0.0:
        return a_s * x_hat + (sig_s / sig_t) * (z_t - a_t * x_hat)

    # Song et al. DDIM eq. (16) noise scale
    sigma_noise = eta * (sig_s / sig_t) * th.sqrt(
        th.clamp(1.0 - (a_t / a_s) ** 2, min=0.0))
    dir_coef = th.sqrt(th.clamp(sig_s ** 2 - sigma_noise ** 2, min=0.0))
    eps_pred = (z_t - a_t * x_hat) / sig_t
    z_s = a_s * x_hat + dir_coef * eps_pred + sigma_noise * th.randn_like(z_t)
    return z_s


@th.no_grad()
def sample(model, n_steps, eta, noise, device, batch_size=32):
    ts = th.linspace(1.0, 0.0, n_steps + 1, device=device)
    out = []
    for i in range(0, noise.shape[0], batch_size):
        z = noise[i:i + batch_size].to(device)
        n = z.shape[0]
        for step in range(n_steps):
            t = th.full((n,), ts[step].item(), device=device)
            s = th.full((n,), ts[step + 1].item(), device=device)
            # never inject noise on the final step
            step_eta = eta if step < n_steps - 1 else 0.0
            x_hat = model.predict_x(z, t)
            z = generalized_step(x_hat, z, t, s, step_eta)
        out.append(z.cpu())
    return th.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    ch_mean = np.load(os.path.join(STATS_DIR, "data_mean.npy")).astype(np.float32)
    ch_std = np.load(os.path.join(STATS_DIR, "data_std.npy")).astype(np.float32)

    test_idx = np.sort(np.load(TEST_IDX))
    with h5py.File(DATA, "r") as f:
        real = f["tensor"][test_idx]
    real_sub = rng.choice(real.reshape(-1), WD_SUB, replace=False)
    real_std = real.std(axis=(0, 2, 3))

    state = th.load(CKPT, map_location="cpu", weights_only=True)
    has_ema = "ema_state_dict" in state
    print(f"checkpoint has EMA: {has_ema}")

    th.manual_seed(SEED)
    noise = th.randn(N_GEN, 4, 128, 128)

    lines = []
    grids = {}
    for weights, kind, steps in CONFIGS:
        if weights == "ema" and not has_ema:
            print(f"[skip] {weights}-{kind}{steps}: no EMA in checkpoint")
            continue
        network = make_unet()
        model = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S,
                                 infer=True)
        key = "ema_state_dict" if weights == "ema" else "model_state_dict"
        model.network.load_state_dict(state[key])
        model.to(device).eval()

        th.manual_seed(SEED + 1)  # same stochastic-path seed across configs
        eta = 1.0 if kind == "ddpm" else 0.0
        s = sample(model, steps, eta, noise.clone(), device)
        gen = s.numpy() * ch_std.reshape(1, -1, 1, 1) + ch_mean.reshape(1, -1, 1, 1)
        del model, network
        th.cuda.empty_cache()

        gsub = rng.choice(gen.reshape(-1), WD_SUB, replace=False)
        wd = float(wasserstein_distance(gsub, real_sub))
        sr = gen.std(axis=(0, 2, 3)) / real_std
        pos = float(((gen[:, 0] <= 0) | (gen[:, 1] <= 0)).mean())
        vx, vy = gen[:, 2], gen[:, 3]
        div = float(np.mean(np.abs(
            (np.roll(vx, -1, axis=2) - np.roll(vx, 1, axis=2)) / 2.0
            + (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / 2.0)))

        label = f"{weights}-{kind}{steps}"
        line = (f"{label:14s} WD={wd:.4f}  pos={pos:.5f}  div={div:.4f}  "
                f"std_ratio=" + " ".join(f"{n}:{sr[c]:.2f}"
                                         for c, n in enumerate(CHANNELS)))
        lines.append(line)
        print(line)
        grids[label] = gen[:4]

    with open(f"{OUT}/sampling_sweep.txt", "w") as fh:
        fh.write("real refs: WD baseline 0.40 | pos 0 | div 0.029\n")
        fh.write("\n".join(lines) + "\n")

    # one comparison grid: real row + one row per config (density + Vy only)
    n_cfg = len(grids)
    fig, axes = plt.subplots(n_cfg + 1, 8, figsize=(16.4, 2.15 * (n_cfg + 1)))
    for j in range(4):
        for c, col in enumerate([0, 3]):
            ax = axes[0, 4 * c + j]
            vmin, vmax = np.percentile(real[:, col], [1, 99])
            ax.imshow(real[j, col], vmin=vmin, vmax=vmax, cmap="RdBu_r")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel("real", fontsize=9)
            ax.set_title(f"{CHANNELS[col]}" if j == 0 else "", fontsize=8)
    for r, (label, g) in enumerate(grids.items(), start=1):
        for j in range(4):
            for c, col in enumerate([0, 3]):
                ax = axes[r, 4 * c + j]
                vmin, vmax = np.percentile(real[:, col], [1, 99])
                ax.imshow(g[j, col], vmin=vmin, vmax=vmax, cmap="RdBu_r")
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(label, fontsize=8)
    plt.suptitle("CNS teacher sampling protocols: density (left 4) and Vy "
                 "(right 4)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/protocol_grid.png", dpi=120, bbox_inches="tight")
    print(f"wrote {OUT}/")


if __name__ == "__main__":
    main()
