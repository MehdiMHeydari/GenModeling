"""
Teacher sampling diagnostic.

Before retraining anything, check whether teacher fuzziness is a sampler issue
or a model issue. Sweeps three axes on the EXISTING teacher:

    1. Checkpoint epoch  (e.g. 200 vs 300 vs 350 vs 399)
    2. DDIM step count   (e.g. 75 vs 150 vs 250)
    3. Sampler order     (Euler/DDIM vs Heun 2nd-order)

For every (ckpt, steps, sampler) combination it:
    - draws samples from a fixed noise seed
    - saves a side-by-side sample grid vs ground truth
    - saves a marginal histogram (generated vs ground truth)
    - records pixel MSE and 1-Wasserstein distance vs a held-out test batch
    - writes one row to diagnostics/teacher_diag.csv

Usage:
    python scripts/diagnose_teacher.py --gpu 7
    python scripts/diagnose_teacher.py --gpu 7 \\
        --ckpts 200 300 350 399 \\
        --step_counts 75 150 250 \\
        --samplers ddim heun \\
        --n_samples 500 \\
        --output_dir diagnostics/teacher_v1
"""

import argparse
import csv
import os

import numpy as np
import h5py
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.diffusion_utils import ddim_step


DATA_SHAPE = (1, 128, 128)
SCHEDULE_S = 0.008
DATA_PATH = "data/2D_DarcyFlow_beta1.0_Train.hdf5"
STATS_DIR = "darcy_teacher/exp_1/saved_state"
CKPT_DIR = "darcy_teacher/exp_1/saved_state"
CMAP = "RdBu_r"

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
def sample_ddim(teacher, noise, n_steps, device, batch_size=64):
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


@th.no_grad()
def sample_heun(teacher, noise, n_steps, device, batch_size=64):
    """
    Heun-style 2nd-order corrector on top of DDIM.
    At each step t -> s:
        x_hat_1 = model(z_t, t)
        z_s_pred = ddim(x_hat_1, z_t, t, s)
        x_hat_2 = model(z_s_pred, s)
        x_hat   = 0.5 * (x_hat_1 + x_hat_2)
        z_s     = ddim(x_hat, z_t, t, s)
    Doubles NFE per step but typically sharper.
    """
    ts = th.linspace(1.0, 0.0, n_steps + 1, device=device)
    out = []
    for i in range(0, noise.shape[0], batch_size):
        z = noise[i:i + batch_size].to(device)
        n = z.shape[0]
        for step in range(n_steps):
            t = th.full((n,), ts[step].item(), device=device)
            s = th.full((n,), ts[step + 1].item(), device=device)
            x_hat_1 = teacher.predict_x(z, t)
            z_pred = ddim_step(x_hat_1, z, t, s, SCHEDULE_S)
            if step < n_steps - 1:
                x_hat_2 = teacher.predict_x(z_pred, s)
                x_hat = 0.5 * (x_hat_1 + x_hat_2)
                z = ddim_step(x_hat, z, t, s, SCHEDULE_S)
            else:
                z = z_pred
        out.append(z.cpu())
    return th.cat(out, dim=0)


def plot_grid(samples, gt, path, title):
    n_show = min(8, samples.shape[0])
    fig, axes = plt.subplots(2, n_show, figsize=(2.2 * n_show, 5.0))
    vmin = min(samples[:n_show].min(), gt[:n_show].min())
    vmax = max(samples[:n_show].max(), gt[:n_show].max())
    for j in range(n_show):
        axes[0, j].imshow(gt[j, 0], cmap=CMAP, vmin=vmin, vmax=vmax)
        axes[0, j].set_title("GT", fontsize=8)
        axes[0, j].axis("off")
        axes[1, j].imshow(samples[j, 0], cmap=CMAP, vmin=vmin, vmax=vmax)
        axes[1, j].set_title("Gen", fontsize=8)
        axes[1, j].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_hist(samples, gt, path, title):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(gt.flatten(), bins=80, density=True, alpha=0.35, label="Ground Truth", color="gray")
    ax.hist(samples.flatten(), bins=80, density=True, histtype="step", linewidth=2, label="Generated")
    ax.set_xlabel("u(x, y)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--ckpts", type=int, nargs="+", default=[200, 300, 350, 399])
    p.add_argument("--step_counts", type=int, nargs="+", default=[75, 150, 250])
    p.add_argument("--samplers", type=str, nargs="+", default=["ddim", "heun"],
                   choices=["ddim", "heun"])
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="diagnostics/teacher_v1")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if th.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    data_min = np.load(os.path.join(STATS_DIR, "data_min.npy"))
    data_max = np.load(os.path.join(STATS_DIR, "data_max.npy"))

    with h5py.File(DATA_PATH, "r") as f:
        data = f["tensor"][:]
    if data.ndim == 3:
        data = data[:, None]
    real_norm = 2.0 * (data - data_min) / (data_max - data_min) - 1.0
    real_batch = real_norm[-args.n_samples:]
    real_denorm = denormalize(real_batch, data_min, data_max)

    th.manual_seed(args.seed)
    initial_noise = th.randn(args.n_samples, *DATA_SHAPE)

    csv_path = os.path.join(args.output_dir, "teacher_diag.csv")
    new_file = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if new_file:
        writer.writerow(["ckpt_epoch", "sampler", "n_steps", "nfe",
                         "pixel_mse", "wasserstein", "mean_err", "std_err"])

    for epoch in args.ckpts:
        print(f"\n=== checkpoint_{epoch}.pt ===")
        teacher = load_teacher(epoch, device)

        for sampler in args.samplers:
            for n_steps in args.step_counts:
                nfe = n_steps if sampler == "ddim" else 2 * n_steps - 1
                tag = f"ckpt{epoch}_{sampler}_{n_steps}steps"
                print(f"  {tag}  (NFE={nfe})")

                if sampler == "ddim":
                    samples = sample_ddim(teacher, initial_noise, n_steps, device)
                else:
                    samples = sample_heun(teacher, initial_noise, n_steps, device)
                gen = denormalize(samples, data_min, data_max)

                pixel_mse = float(((gen - real_denorm) ** 2).mean())
                wd = float(wasserstein_distance(gen.flatten()[:50000],
                                                real_denorm.flatten()[:50000]))
                mean_err = float(abs(gen.mean() - real_denorm.mean()))
                std_err = float(abs(gen.std() - real_denorm.std()))

                writer.writerow([epoch, sampler, n_steps, nfe,
                                 pixel_mse, wd, mean_err, std_err])
                csv_file.flush()

                title = f"ckpt {epoch}  |  {sampler.upper()} {n_steps} steps  |  NFE={nfe}"
                plot_grid(gen, real_denorm,
                          os.path.join(args.output_dir, f"{tag}_grid.png"), title)
                plot_hist(gen, real_denorm,
                          os.path.join(args.output_dir, f"{tag}_hist.png"), title)

        del teacher
        th.cuda.empty_cache()

    csv_file.close()
    print(f"\nDone. Results in {args.output_dir}/teacher_diag.csv")


if __name__ == "__main__":
    main()
