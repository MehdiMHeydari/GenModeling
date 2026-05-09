"""
RD teacher sampling diagnostic.

Mirror of scripts/diagnose_ns_teacher.py but for the RD teacher (also
2-channel — u and v concentrations from FitzHugh-Nagumo). Sweeps:
    1. Checkpoint epoch  (e.g. 75, 200, 599)
    2. DDIM step count   (75 worked well on NS — start there)

Outputs to diagnostics/rd_teacher_v1/:
    - per-combo sample grid (8 GT vs 8 Gen, magnitude field)
    - per-combo single-channel histogram (overlay on each channel)
    - per-combo magnitude histogram
    - rd_teacher_diag.csv with pixel_mse, wasserstein, mean_err, std_err

Usage:
    python scripts/diagnose_rd_teacher.py --gpu 6
    python scripts/diagnose_rd_teacher.py --gpu 6 \\
        --ckpts 75 200 400 599 --step_counts 75 250 --n_samples 200
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

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.diffusion_utils import ddim_step


DATA_SHAPE = (2, 128, 128)
SCHEDULE_S = 0.008
DATA_PATH = "data/rd_128_merged.h5"
STATS_DIR = "rd_teacher/exp_1/saved_state"
CKPT_DIR = "rd_teacher/exp_1/saved_state"

UNET_CFG = dict(
    dim=list(DATA_SHAPE),
    channel_mult="1, 2, 4, 4",
    num_channels=64,
    num_res_blocks=2,
    num_head_channels=32,
    attention_resolutions="32",
    dropout=0.0,
    use_new_attention_order=True,
    use_scale_shift_norm=True,
    class_cond=False,
    num_classes=None,
)


def denormalize(samples, data_min, data_max):
    if isinstance(samples, th.Tensor):
        samples = samples.cpu().numpy()
    return (samples + 1.0) / 2.0 * (data_max - data_min) + data_min


def magnitude(field):
    """field: (N, 2, H, W) -> (N, H, W) sqrt(u^2 + v^2). Same reduction as NS |v|.
    For RD the channels are (u, v) concentrations from FitzHugh-Nagumo, so this
    is just the L2 norm of the channel pair — physical meaning is "total
    activity" rather than velocity speed, but the metric is the same."""
    return np.sqrt(field[:, 0] ** 2 + field[:, 1] ** 2)


def load_teacher(ckpt_epoch, device):
    ckpt_path = os.path.join(CKPT_DIR, f"checkpoint_{ckpt_epoch}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing teacher checkpoint: {ckpt_path}")
    network = UNetModel(**UNET_CFG)
    teacher = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(ckpt_path, map_location="cpu", weights_only=True)
    teacher.network.load_state_dict(state["model_state_dict"])
    teacher.to(device).eval()
    return teacher


@th.no_grad()
def sample_ddim(teacher, noise, n_steps, device, batch_size=32):
    ts = th.linspace(1.0, 0.0, n_steps + 1, device=device)
    out = []
    for i in range(0, noise.shape[0], batch_size):
        z = noise[i:i + batch_size].to(device)
        n = z.shape[0]
        for step in range(n_steps):
            t = th.full((n,), ts[step].item(), device=device)
            s = th.full((n,), ts[step + 1].item(), device=device)
            x_hat = teacher.predict_x(z, t)
            z = ddim_step(x_hat, z, t, s, SCHEDULE_S)
        out.append(z.cpu())
    return th.cat(out, dim=0)


def plot_grid(samples_denorm, gt_denorm, path, title):
    n_show = min(8, samples_denorm.shape[0])
    gt_mag = magnitude(gt_denorm[:n_show])
    gen_mag = magnitude(samples_denorm[:n_show])
    fig, axes = plt.subplots(2, n_show, figsize=(2.2 * n_show, 5.0))
    for j in range(n_show):
        axes[0, j].imshow(gt_mag[j])
        axes[0, j].set_title("GT", fontsize=8)
        axes[0, j].axis("off")
        axes[1, j].imshow(gen_mag[j])
        axes[1, j].set_title("Gen", fontsize=8)
        axes[1, j].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_hist(samples_denorm, gt_denorm, path, title):
    """Magnitude (sqrt(u² + v²)) histogram."""
    gt_m = magnitude(gt_denorm).flatten()
    gen_m = magnitude(samples_denorm).flatten()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(gt_m, bins=80, density=True, alpha=0.35, label="Ground Truth", color="gray")
    ax.hist(gen_m, bins=80, density=True, histtype="step", linewidth=2, label="Generated")
    ax.set_xlabel("sqrt(u² + v²)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_hist_per_channel(samples_denorm, gt_denorm, path, title):
    """Per-channel u and v histograms — direct (signed) comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ch, name in enumerate(["u", "v"]):
        gt_ch = gt_denorm[:, ch].flatten()
        gen_ch = samples_denorm[:, ch].flatten()
        ax = axes[ch]
        ax.hist(gt_ch, bins=80, density=True, alpha=0.35, label="Ground Truth", color="gray")
        ax.hist(gen_ch, bins=80, density=True, histtype="step", linewidth=2, label="Generated")
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(name)
        ax.legend()
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--ckpts", type=int, nargs="+",
                   default=[75, 200, 400, 599])
    p.add_argument("--step_counts", type=int, nargs="+", default=[75])
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="diagnostics/rd_teacher_v1")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if th.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    data_min = np.load(os.path.join(STATS_DIR, "data_min.npy"))
    data_max = np.load(os.path.join(STATS_DIR, "data_max.npy"))

    with h5py.File(DATA_PATH, "r") as f:
        data = f["tensor"][:]
    real_norm = 2.0 * (data - data_min) / (data_max - data_min) - 1.0
    rng = np.random.RandomState(args.seed)
    gt_indices = rng.choice(len(real_norm), size=args.n_samples, replace=False)
    real_batch = real_norm[gt_indices]
    real_denorm = denormalize(real_batch, data_min, data_max)

    th.manual_seed(args.seed)
    initial_noise = th.randn(args.n_samples, *DATA_SHAPE)

    csv_path = os.path.join(args.output_dir, "rd_teacher_diag.csv")
    new_file = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if new_file:
        writer.writerow(["ckpt_epoch", "n_steps", "nfe", "pixel_mse",
                         "wasserstein_mag", "mean_err", "std_err"])

    for epoch in args.ckpts:
        print(f"\n=== checkpoint_{epoch}.pt ===")
        teacher = load_teacher(epoch, device)

        for n_steps in args.step_counts:
            tag = f"ckpt{epoch}_ddim_{n_steps}steps"
            print(f"  {tag}  (NFE={n_steps})")

            samples = sample_ddim(teacher, initial_noise, n_steps, device)
            gen = denormalize(samples, data_min, data_max)

            pixel_mse = float(((gen - real_denorm) ** 2).mean())
            gen_m = magnitude(gen).flatten()
            real_m = magnitude(real_denorm).flatten()
            cap = min(len(gen_m), len(real_m), 50000)
            sub_rng = np.random.RandomState(0)
            gs = gen_m[sub_rng.choice(len(gen_m), size=cap, replace=False)]
            rs = real_m[sub_rng.choice(len(real_m), size=cap, replace=False)]
            wd = float(wasserstein_distance(gs, rs))
            mean_err = float(abs(gen.mean() - real_denorm.mean()))
            std_err = float(abs(gen.std() - real_denorm.std()))

            writer.writerow([epoch, n_steps, n_steps,
                             pixel_mse, wd, mean_err, std_err])
            csv_file.flush()
            print(f"    WD={wd:.4f}  pixel_mse={pixel_mse:.4f}  "
                  f"std_err={std_err:.4f}")

            title = f"RD ckpt {epoch}  |  DDIM {n_steps} steps  |  NFE={n_steps}"
            plot_grid(gen, real_denorm,
                      os.path.join(args.output_dir, f"{tag}_grid.png"), title)
            plot_hist(gen, real_denorm,
                      os.path.join(args.output_dir, f"{tag}_hist.png"), title)
            plot_hist_per_channel(gen, real_denorm,
                                  os.path.join(args.output_dir, f"{tag}_hist_uv.png"), title)

        del teacher
        th.cuda.empty_cache()

    csv_file.close()
    print(f"\nDone. Results in {args.output_dir}/rd_teacher_diag.csv")


if __name__ == "__main__":
    main()
