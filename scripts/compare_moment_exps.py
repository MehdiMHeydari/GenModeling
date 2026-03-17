"""
Compare moment-matching CD experiments (exp 6, 7, 9) against baseline (exp 3),
teacher, and ground truth. Same sampling approach as compare_cd_histogram.py.

Produces:
  1. Sample grid: rows = ground truth, teacher, baseline, moment exps
  2. Histogram: pixel distribution overlaid + line

Usage:
    python scripts/compare_moment_exps.py --gpu 0 --n_samples 1000
"""

import argparse
import glob
import os
import re
import numpy as np
import h5py
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.consistency_models import MultistepConsistencyModel
from src.inference.samplers import MultistepCMSampler
from src.models.diffusion_utils import ddim_step


UNET_CFG = dict(
    dim=[1, 128, 128],
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

DATA_SHAPE = (1, 128, 128)
SCHEDULE_S = 0.008
STUDENT_STEPS = 16
DDIM_STEPS = 50
DATA_PATH = "data/2D_DarcyFlow_beta1.0_Train.hdf5"
STATS_DIR = "darcy_teacher/exp_1/saved_state"
TEACHER_CKPT = "darcy_teacher/exp_1/saved_state/checkpoint_200.pt"


def find_latest_checkpoint(exp_dir):
    pattern = os.path.join(exp_dir, "saved_state", "checkpoint_*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    def epoch_num(path):
        m = re.search(r"checkpoint_(\d+)\.pt", path)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=epoch_num)


def load_and_sample_cd(ckpt_path, initial_noise, device):
    network = UNetModel(**UNET_CFG)
    model = MultistepConsistencyModel(
        network=network,
        student_steps=STUDENT_STEPS,
        schedule_s=SCHEDULE_S,
        infer=True,
    )
    state = th.load(ckpt_path, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    if "ema_state_dict" in state:
        model.ema_network.load_state_dict(state["ema_state_dict"])
    model.to(device).eval()

    sampler = MultistepCMSampler(model)
    batch_size = 64
    all_samples = []
    with th.no_grad():
        for i in range(0, initial_noise.shape[0], batch_size):
            z = initial_noise[i:i+batch_size].to(device)
            samples = sampler.sample(z)
            all_samples.append(samples.cpu())

    del model, network
    th.cuda.empty_cache()
    return th.cat(all_samples, dim=0)


def load_and_sample_teacher(initial_noise, device):
    network = UNetModel(**UNET_CFG)
    teacher = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(TEACHER_CKPT, map_location="cpu", weights_only=True)
    if "ema_state_dict" in state:
        teacher.network.load_state_dict(state["ema_state_dict"])
    else:
        teacher.network.load_state_dict(state["model_state_dict"])
    teacher.to(device).eval()

    ts = th.linspace(1.0, 0.0, DDIM_STEPS + 1, device=device)
    batch_size = 64
    all_samples = []

    with th.no_grad():
        for i in range(0, initial_noise.shape[0], batch_size):
            z = initial_noise[i:i+batch_size].to(device)
            n = z.shape[0]
            for step in range(DDIM_STEPS):
                t_batch = th.full((n,), ts[step].item(), device=device)
                s_batch = th.full((n,), ts[step + 1].item(), device=device)
                x_hat = teacher.predict_x(z, t_batch)
                z = ddim_step(x_hat, z, t_batch, s_batch, SCHEDULE_S)
            all_samples.append(z.cpu())

    del teacher, network
    th.cuda.empty_cache()
    return th.cat(all_samples, dim=0)


def denormalize(samples, data_min, data_max):
    if isinstance(samples, th.Tensor):
        samples = samples.numpy()
    return (samples + 1.0) / 2.0 * (data_max - data_min) + data_min


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of samples for histogram (use 1000+ for smooth curves)")
    parser.add_argument("--n_show", type=int, default=6,
                        help="Number of samples to show in the visual grid")
    parser.add_argument("--output_dir", type=str, default="eval_moment_ddim")
    args = parser.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load norm stats
    data_min = float(np.load(os.path.join(STATS_DIR, "data_min.npy")))
    data_max = float(np.load(os.path.join(STATS_DIR, "data_max.npy")))

    # Load ground truth
    with h5py.File(DATA_PATH, "r") as f:
        outputs = np.array(f["tensor"]).astype(np.float32)
    if outputs.ndim == 3:
        outputs = outputs[:, np.newaxis, :, :]
    real_denorm = outputs[9500:]  # test set
    print(f"Ground truth: {real_denorm.shape[0]} test samples")

    # Fixed noise for all models
    th.manual_seed(42)
    initial_noise = th.randn(args.n_samples, *DATA_SHAPE)

    # --- Experiments ---
    exps = {
        "exp_3": {"dir": "darcy_student/exp_3",
                  "label": "CD baseline\n(no moment)"},
        "exp_6": {"dir": "darcy_student/exp_6",
                  "label": "Moment exp6\n(var=0.1)"},
        "exp_7": {"dir": "darcy_student/exp_7",
                  "label": "Moment exp7\n(var=1.0)"},
        "exp_9": {"dir": "darcy_student/exp_9",
                  "label": "Moment exp9\n(mu=0.1, var=1.0)"},
    }

    all_samples = {}
    for name, info in exps.items():
        ckpt = find_latest_checkpoint(info["dir"])
        if ckpt is None:
            print(f"WARNING: No checkpoint found for {name}, skipping")
            continue
        epoch = re.search(r"checkpoint_(\d+)", ckpt).group(1)
        info["epoch"] = epoch
        print(f"Sampling {name} (epoch {epoch})...")
        samples = load_and_sample_cd(ckpt, initial_noise, device)
        all_samples[name] = denormalize(samples, data_min, data_max)

    # Teacher
    print("Sampling teacher (50 aDDIM steps)...")
    teacher_samples = load_and_sample_teacher(initial_noise, device)
    teacher_denorm = denormalize(teacher_samples, data_min, data_max)

    # ============================================================
    # PLOT 1: Side-by-side sample grid
    # ============================================================
    active_exps = [k for k in exps if k in all_samples]
    n_rows = len(active_exps) + 2  # +teacher +ground truth
    n_cols = args.n_show

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.5 * n_cols + 1.5, 2.5 * n_rows))
    plt.subplots_adjust(left=0.12, wspace=0.05, hspace=0.15)

    row = 0
    # Ground truth
    for j in range(n_cols):
        axes[row, j].imshow(real_denorm[j, 0], cmap="RdBu_r")
        axes[row, j].axis("off")
    axes[row, 0].text(-0.15, 0.5, "Ground Truth",
                      transform=axes[row, 0].transAxes, fontsize=11,
                      fontweight="bold", va="center", ha="right")
    row += 1

    # Teacher
    for j in range(n_cols):
        axes[row, j].imshow(teacher_denorm[j, 0], cmap="RdBu_r")
        axes[row, j].axis("off")
    axes[row, 0].text(-0.15, 0.5, "Teacher\n(50 DDIM)",
                      transform=axes[row, 0].transAxes, fontsize=11,
                      fontweight="bold", va="center", ha="right")
    row += 1

    # CD experiments
    for name in active_exps:
        info = exps[name]
        for j in range(n_cols):
            axes[row, j].imshow(all_samples[name][j, 0], cmap="RdBu_r")
            axes[row, j].axis("off")
        axes[row, 0].text(-0.15, 0.5, f"{info['label']}\n(ep {info['epoch']})",
                          transform=axes[row, 0].transAxes, fontsize=10,
                          fontweight="bold", va="center", ha="right")
        row += 1

    fig.suptitle("CD Moment-Matching: Sample Comparison (16-step sampling)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    grid_path = os.path.join(args.output_dir, "moment_samples.png")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {grid_path}")

    # ============================================================
    # PLOT 2: Histogram — overlaid + line
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    colors = {
        "Ground Truth": "gray",
        "Teacher (50 DDIM)": "tab:blue",
        "CD baseline (no moment)": "tab:red",
        "Moment exp6 (var=0.1)": "tab:orange",
        "Moment exp7 (var=1.0)": "tab:green",
        "Moment exp9 (mu=0.1, var=1.0)": "tab:purple",
    }

    hist_data = {"Ground Truth": real_denorm.flatten(),
                 "Teacher (50 DDIM)": teacher_denorm.flatten()}
    label_map = {
        "exp_3": "CD baseline (no moment)",
        "exp_6": "Moment exp6 (var=0.1)",
        "exp_7": "Moment exp7 (var=1.0)",
        "exp_9": "Moment exp9 (mu=0.1, var=1.0)",
    }
    for name in active_exps:
        hist_data[label_map[name]] = all_samples[name].flatten()

    bins = np.linspace(0, real_denorm.max() * 1.05, 150)

    # Left: overlaid filled histograms
    for label, data in hist_data.items():
        ax1.hist(data, bins=bins, alpha=0.3, density=True,
                 label=label, color=colors[label])
    ax1.set_xlabel("u(x, y)", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Pixel Value Distribution (Overlaid)", fontsize=13)
    ax1.legend(fontsize=9)

    # Right: line histograms
    for label, data in hist_data.items():
        counts, edges = np.histogram(data, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax2.plot(centers, counts, label=label, color=colors[label], linewidth=1.5)
    ax2.set_xlabel("u(x, y)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Pixel Value Distribution (Line)", fontsize=13)
    ax2.legend(fontsize=9)

    fig.suptitle("Darcy Flow: Moment-Matching Regularization Effect",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    hist_path = os.path.join(args.output_dir, "moment_histogram.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {hist_path}")

    # ============================================================
    # STATS
    # ============================================================
    print("\n" + "=" * 75)
    print("STATISTICS (physical units)")
    print("=" * 75)
    header = f"{'':30s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}"
    print(header)
    print("-" * 75)
    for label, data in hist_data.items():
        print(f"{label:30s} {data.mean():10.4f} {data.std():10.4f} "
              f"{data.min():10.4f} {data.max():10.4f}")

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
