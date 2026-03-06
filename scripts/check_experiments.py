"""
Check all running experiments: sample from latest checkpoint of each
and plot a labeled side-by-side comparison grid.

Experiments checked:
  - Teacher (50-step DDIM) — reference
  - CD exp_5 (Consistency Distillation, grad_accum=8)
  - MFM (Mean Flow Matching)
  - PD (Progressive Distillation)

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/check_experiments.py --gpu 0 --n_samples 8
"""

import argparse
import glob
import os
import re
import torch as th
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.consistency_models import MultistepConsistencyModel
from src.models.flow_models import MeanFlowMatching
from src.models.diffusion_utils import ddim_step
from src.inference.samplers import MultistepCMSampler, MeanSampler


# ============================================================
# Config
# ============================================================
DATA_SHAPE = (1, 128, 128)
SCHEDULE_S = 0.008
DDIM_STEPS = 50

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

TEACHER_CKPT = "darcy_teacher/exp_1/saved_state/checkpoint_200.pt"
NORM_DIR = "darcy_teacher/exp_1/saved_state"


# ============================================================
# Checkpoint finders
# ============================================================

def find_latest_epoch_checkpoint(save_dir):
    """Find checkpoint_*.pt with the highest epoch number."""
    if not os.path.isdir(save_dir):
        return None, None
    ckpts = glob.glob(os.path.join(save_dir, "checkpoint_*.pt"))
    if not ckpts:
        return None, None

    def epoch_num(path):
        m = re.search(r"checkpoint_(\d+)\.pt", path)
        return int(m.group(1)) if m else -1

    best = max(ckpts, key=epoch_num)
    return best, epoch_num(best)


def find_latest_pd_checkpoint(save_dir):
    """Find pd_round*_steps*.pt with the highest round number."""
    if not os.path.isdir(save_dir):
        return None, None, None
    ckpts = glob.glob(os.path.join(save_dir, "pd_round*_steps*.pt"))
    if not ckpts:
        return None, None, None

    def round_num(path):
        m = re.search(r"pd_round(\d+)_steps(\d+)\.pt", path)
        return int(m.group(1)) if m else -1

    best = max(ckpts, key=round_num)
    m = re.search(r"pd_round(\d+)_steps(\d+)\.pt", best)
    return best, int(m.group(1)), int(m.group(2))


# ============================================================
# Denormalization
# ============================================================

def load_norm_stats():
    min_path = os.path.join(NORM_DIR, "data_min.npy")
    max_path = os.path.join(NORM_DIR, "data_max.npy")
    if os.path.exists(min_path) and os.path.exists(max_path):
        return float(np.load(min_path)), float(np.load(max_path))
    return None, None


def denormalize(samples, data_min, data_max):
    if data_min is not None and data_max is not None:
        return (samples + 1.0) / 2.0 * (data_max - data_min) + data_min
    return samples


# ============================================================
# Sampling functions
# ============================================================

def sample_teacher(device, z):
    """Load teacher and sample with plain DDIM."""
    assert os.path.exists(TEACHER_CKPT), f"No teacher checkpoint at {TEACHER_CKPT}"

    network = UNetModel(**UNET_CFG)
    model = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(TEACHER_CKPT, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Teacher: {TEACHER_CKPT}")

    ts = th.linspace(1.0, 0.0, DDIM_STEPS + 1, device=device)
    x = z.clone().to(device)
    n = x.shape[0]

    with th.no_grad():
        for i in range(DDIM_STEPS):
            t_batch = th.full((n,), ts[i].item(), device=device)
            s_batch = th.full((n,), ts[i + 1].item(), device=device)
            x_hat = model.predict_x(x, t_batch)
            x = ddim_step(x_hat, x, t_batch, s_batch, SCHEDULE_S)

    return x.cpu(), f"Teacher\n({DDIM_STEPS} DDIM steps)"


def sample_cd_exp5(device, z):
    """Load CD exp_5 latest checkpoint and sample."""
    save_dir = "darcy_student/exp_5/saved_state"
    ckpt, epoch = find_latest_epoch_checkpoint(save_dir)
    if ckpt is None:
        print("  CD exp_5: no checkpoints found, skipping")
        return None, None

    print(f"  CD exp_5: {ckpt} (epoch {epoch})")

    network = UNetModel(**UNET_CFG)
    model = MultistepConsistencyModel(
        network=network, student_steps=16, schedule_s=SCHEDULE_S, infer=True,
    )
    state = th.load(ckpt, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    if "ema_state_dict" in state:
        model.ema_network.load_state_dict(state["ema_state_dict"])
    model.to(device)
    model.eval()

    sampler = MultistepCMSampler(model)
    with th.no_grad():
        samples = sampler.sample(z.to(device))

    return samples.cpu(), f"CD exp_5\n(epoch {epoch}, 16 steps)"


def sample_mfm(device, z):
    """Load MFM latest checkpoint and sample."""
    save_dir = "darcy_mean_flow/exp_2/saved_state"
    ckpt, epoch = find_latest_epoch_checkpoint(save_dir)
    if ckpt is None:
        print("  MFM: no checkpoints found, skipping")
        return None, None

    print(f"  MFM: {ckpt} (epoch {epoch})")

    unet_cfg = dict(UNET_CFG)
    unet_cfg["use_future_time_emb"] = True
    network = UNetModel(**unet_cfg)
    model = MeanFlowMatching(network=network, infer=True)
    state = th.load(ckpt, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    sampler = MeanSampler(model)
    with th.no_grad():
        samples = sampler.sample(z.to(device))

    return samples.cpu(), f"MFM\n(epoch {epoch}, 1 step)"


def sample_pd(device, z):
    """Load PD latest round checkpoint and sample with DDIM."""
    save_dir = "darcy_pd/exp_1/saved_state"
    ckpt, round_num, student_steps = find_latest_pd_checkpoint(save_dir)
    if ckpt is None:
        print("  PD: no checkpoints found, skipping")
        return None, None

    print(f"  PD: {ckpt} (round {round_num}, {student_steps} steps)")

    network = UNetModel(**UNET_CFG)
    model = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(ckpt, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    # DDIM sampling with student_steps
    ts = th.linspace(1.0, 0.0, student_steps + 1, device=device)
    x = z.clone().to(device)
    n = x.shape[0]

    with th.no_grad():
        for i in range(student_steps):
            t_batch = th.full((n,), ts[i].item(), device=device).clamp(1e-4, 1 - 1e-4)
            s_batch = th.full((n,), ts[i + 1].item(), device=device).clamp(0, 1 - 1e-4)
            x_hat = model.predict_x(x, t_batch, use_ema=True)
            x = ddim_step(x_hat, x, t_batch, s_batch, SCHEDULE_S)

    return x.cpu(), f"PD\n(round {round_num}, {student_steps} steps)"


# ============================================================
# Plotting
# ============================================================

def plot_grid(results, n_samples, save_path):
    """Plot a labeled grid: one row per model, n_samples columns."""
    # Filter out None results
    results = [(samples, label) for samples, label in results if samples is not None]

    if not results:
        print("No experiments have checkpoints yet — nothing to plot.")
        return

    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(2.5 * n_samples, 3 * n_rows + 0.5))

    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_samples == 1:
        axes = axes[:, np.newaxis]

    # Column labels
    for j in range(n_samples):
        axes[0, j].set_title(f"Sample {j+1}", fontsize=10, fontweight="bold")

    # Plot each row
    for i, (samples, label) in enumerate(results):
        for j in range(n_samples):
            img = samples[j, 0].numpy()
            axes[i, j].imshow(img, cmap="RdBu_r")
            axes[i, j].axis("off")

        # Row label on the left
        axes[i, 0].set_ylabel(label, fontsize=11, fontweight="bold",
                               rotation=0, labelpad=80, va="center", ha="right")

    fig.suptitle("Experiment Check", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.08, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="experiment_check.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    n = args.n_samples

    # Shared noise for fair comparison
    th.manual_seed(args.seed)
    z = th.randn(n, *DATA_SHAPE)

    # Load norm stats
    data_min, data_max = load_norm_stats()

    print("Loading and sampling from each experiment...\n")

    results = []

    # 1. Teacher
    samples, label = sample_teacher(device, z)
    samples = denormalize(samples, data_min, data_max)
    results.append((samples, label))

    # 2. CD exp_5
    samples, label = sample_cd_exp5(device, z)
    if samples is not None:
        samples = denormalize(samples, data_min, data_max)
    results.append((samples, label))

    # 3. MFM
    samples, label = sample_mfm(device, z)
    if samples is not None:
        samples = denormalize(samples, data_min, data_max)
    results.append((samples, label))

    # 4. PD
    samples, label = sample_pd(device, z)
    if samples is not None:
        samples = denormalize(samples, data_min, data_max)
    results.append((samples, label))

    # Plot
    plot_grid(results, n, args.save_path)


if __name__ == "__main__":
    main()
