"""
Side-by-side comparison of Progressive Distillation vs Consistency Distillation
at matched step counts (16, 8, 4).

Rows: Teacher | PD-16 | CD-16 | PD-8 | CD-8 | PD-4 | CD-4

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/compare_pd_vs_cd.py --gpu 0 --n_samples 8
"""

import argparse
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
from src.models.diffusion_utils import ddim_step
from src.inference.samplers import MultistepCMSampler


# ============================================================
# Config
# ============================================================
DATA_SHAPE = (1, 128, 128)
SCHEDULE_S = 0.008

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

# PD checkpoints: step_count -> checkpoint path
PD_DIR = "darcy_pd/exp_1/saved_state"

# CD experiments: step_count -> (exp_dir, student_steps)
CD_EXPERIMENTS = {
    16: "darcy_student/exp_5/saved_state",   # best 16-step (grad_accum=8)
    8:  "darcy_student/exp_2/saved_state",   # 8-step
    4:  "darcy_student/exp_1/saved_state",   # 4-step (mode collapsed)
}


# ============================================================
# Helpers
# ============================================================

def find_latest_epoch_checkpoint(save_dir):
    import glob
    if not os.path.isdir(save_dir):
        return None, None
    ckpts = glob.glob(os.path.join(save_dir, "checkpoint_*.pt"))
    if not ckpts:
        return None, None
    def epoch_num(p):
        m = re.search(r"checkpoint_(\d+)\.pt", p)
        return int(m.group(1)) if m else -1
    best = max(ckpts, key=epoch_num)
    return best, epoch_num(best)


def find_pd_checkpoint(save_dir, target_steps):
    """Find PD checkpoint for a specific step count."""
    import glob
    pattern = os.path.join(save_dir, f"pd_round*_steps{target_steps}.pt")
    matches = glob.glob(pattern)
    if not matches:
        return None, None
    # Take the one with highest round number
    def round_num(p):
        m = re.search(r"pd_round(\d+)_steps", p)
        return int(m.group(1)) if m else -1
    best = max(matches, key=round_num)
    return best, round_num(best)


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
# Sampling
# ============================================================

def sample_teacher(device, z):
    network = UNetModel(**UNET_CFG)
    model = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(TEACHER_CKPT, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    ts = th.linspace(1.0, 0.0, 51, device=device)
    x = z.clone().to(device)
    n = x.shape[0]

    with th.no_grad():
        for i in range(50):
            t_batch = th.full((n,), ts[i].item(), device=device)
            s_batch = th.full((n,), ts[i + 1].item(), device=device)
            x_hat = model.predict_x(x, t_batch)
            x = ddim_step(x_hat, x, t_batch, s_batch, SCHEDULE_S)

    return x.cpu()


def sample_pd(device, z, target_steps):
    ckpt, round_num = find_pd_checkpoint(PD_DIR, target_steps)
    if ckpt is None:
        print(f"  PD {target_steps}-step: no checkpoint found, skipping")
        return None

    print(f"  PD {target_steps}-step: {ckpt} (round {round_num})")

    network = UNetModel(**UNET_CFG)
    model = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(ckpt, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    ts = th.linspace(1.0, 0.0, target_steps + 1, device=device)
    x = z.clone().to(device)
    n = x.shape[0]

    with th.no_grad():
        for i in range(target_steps):
            t_batch = th.full((n,), ts[i].item(), device=device).clamp(1e-4, 1 - 1e-4)
            s_batch = th.full((n,), ts[i + 1].item(), device=device).clamp(0, 1 - 1e-4)
            x_hat = model.predict_x(x, t_batch, use_ema=True)
            x = ddim_step(x_hat, x, t_batch, s_batch, SCHEDULE_S)

    return x.cpu()


def sample_cd(device, z, target_steps, save_dir):
    ckpt, epoch = find_latest_epoch_checkpoint(save_dir)
    if ckpt is None:
        print(f"  CD {target_steps}-step: no checkpoint in {save_dir}, skipping")
        return None

    print(f"  CD {target_steps}-step: {ckpt} (epoch {epoch})")

    network = UNetModel(**UNET_CFG)
    model = MultistepConsistencyModel(
        network=network, student_steps=target_steps, schedule_s=SCHEDULE_S, infer=True,
    )
    state = th.load(ckpt, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    if "ema_state_dict" in state:
        model.ema_network.load_state_dict(state["ema_state_dict"])
    model.to(device).eval()

    sampler = MultistepCMSampler(model)
    with th.no_grad():
        samples = sampler.sample(z.to(device))

    return samples.cpu()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="pd_vs_cd_comparison.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    n = args.n_samples

    th.manual_seed(args.seed)
    z = th.randn(n, *DATA_SHAPE)

    data_min, data_max = load_norm_stats()

    print("Loading and sampling...\n")

    # Collect rows: (samples_tensor, label)
    rows = []

    # Teacher
    print("  Teacher: 50-step DDIM")
    teacher_samples = sample_teacher(device, z)
    rows.append((denormalize(teacher_samples, data_min, data_max),
                 "Teacher\n(50 DDIM steps)"))

    # For each step count, add PD then CD
    for step_count in [16, 8, 4]:
        # PD
        pd_samples = sample_pd(device, z, step_count)
        if pd_samples is not None:
            rows.append((denormalize(pd_samples, data_min, data_max),
                         f"PD\n({step_count} steps)"))

        # CD
        cd_dir = CD_EXPERIMENTS[step_count]
        cd_samples = sample_cd(device, z, step_count, cd_dir)
        if cd_samples is not None:
            rows.append((denormalize(cd_samples, data_min, data_max),
                         f"CD\n({step_count} steps)"))

    # Plot
    if len(rows) <= 1:
        print("Not enough experiments to compare.")
        return

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n, figsize=(2.5 * n, 3 * n_rows + 0.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for j in range(n):
        axes[0, j].set_title(f"Sample {j+1}", fontsize=10, fontweight="bold")

    for i, (samples, label) in enumerate(rows):
        for j in range(n):
            axes[i, j].imshow(samples[j, 0].numpy(), cmap="RdBu_r")
            axes[i, j].axis("off")

    # Use fig.text for row labels so they don't get clipped
    plt.tight_layout(rect=[0.12, 0, 1, 0.95])
    for i, (_, label) in enumerate(rows):
        # Get the vertical center of each row in figure coords
        bbox = axes[i, 0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(0.01, y_center, label, fontsize=11, fontweight="bold",
                 va="center", ha="left")

    fig.suptitle("Progressive Distillation vs Consistency Distillation",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(args.save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.save_path}")


if __name__ == "__main__":
    main()
