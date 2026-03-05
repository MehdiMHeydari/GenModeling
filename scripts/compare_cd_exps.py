"""
Compare CD exp_4 vs exp_5: sample from the latest checkpoint of each
and plot a side-by-side grid.

Usage:
    python scripts/compare_cd_exps.py [--gpu 2] [--n_samples 8]
"""

import argparse
import glob
import os
import re
import torch as th
import numpy as np
import matplotlib.pyplot as plt

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.consistency_models import MultistepConsistencyModel
from src.inference.samplers import MultistepCMSampler


UNET_KWARGS = dict(
    dim=[1, 128, 128],
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
)

SCHEDULE_S = 0.008
STUDENT_STEPS = 16
SHAPE = (1, 128, 128)


def find_latest_checkpoint(exp_dir):
    """Find the checkpoint with the highest epoch number."""
    pattern = os.path.join(exp_dir, "saved_state", "checkpoint_*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    # Extract epoch number and sort
    def epoch_num(path):
        m = re.search(r"checkpoint_(\d+)\.pt", path)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=epoch_num)


def load_and_sample(ckpt_path, n_samples, device):
    """Load a CD model from checkpoint and generate samples."""
    network = UNetModel(**UNET_KWARGS)
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

    model.to(device)
    model.eval()

    sampler = MultistepCMSampler(model)

    th.manual_seed(42)
    z = th.randn(n_samples, *SHAPE, device=device)
    with th.no_grad():
        samples = sampler.sample(z)

    return samples.cpu()


def denormalize(samples, stats_dir):
    """Denormalize from [-1, 1] to original range."""
    min_path = os.path.join(stats_dir, "data_min.npy")
    max_path = os.path.join(stats_dir, "data_max.npy")
    if os.path.exists(min_path) and os.path.exists(max_path):
        data_min = float(np.load(min_path))
        data_max = float(np.load(max_path))
        return (samples + 1.0) / 2.0 * (data_max - data_min) + data_min
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="cd_comparison.png")
    args = parser.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    n = args.n_samples

    stats_dir = "darcy_teacher/exp_1/saved_state"

    # Find latest checkpoints
    exp4_ckpt = find_latest_checkpoint("darcy_student/exp_4")
    exp5_ckpt = find_latest_checkpoint("darcy_student/exp_5")

    assert exp4_ckpt, "No checkpoints found for exp_4"
    assert exp5_ckpt, "No checkpoints found for exp_5"

    exp4_epoch = re.search(r"checkpoint_(\d+)", exp4_ckpt).group(1)
    exp5_epoch = re.search(r"checkpoint_(\d+)", exp5_ckpt).group(1)

    print(f"Exp 4: {exp4_ckpt} (epoch {exp4_epoch})")
    print(f"Exp 5: {exp5_ckpt} (epoch {exp5_epoch})")

    # Sample from both
    print("Sampling from exp_4...")
    samples_4 = load_and_sample(exp4_ckpt, n, device)
    samples_4 = denormalize(samples_4, stats_dir)

    print("Sampling from exp_5...")
    samples_5 = load_and_sample(exp5_ckpt, n, device)
    samples_5 = denormalize(samples_5, stats_dir)

    # Plot side-by-side grid
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.5))

    for j in range(n):
        img4 = samples_4[j, 0].numpy()
        img5 = samples_5[j, 0].numpy()

        axes[0, j].imshow(img4, cmap="RdBu_r")
        axes[0, j].axis("off")

        axes[1, j].imshow(img5, cmap="RdBu_r")
        axes[1, j].axis("off")

    axes[0, 0].set_ylabel(f"Exp 4 (ep {exp4_epoch})\ngrad_accum=4", fontsize=11)
    axes[1, 0].set_ylabel(f"Exp 5 (ep {exp5_epoch})\ngrad_accum=8", fontsize=11)

    fig.suptitle("CD Exp 4 vs Exp 5", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison to {args.save_path}")


if __name__ == "__main__":
    main()
