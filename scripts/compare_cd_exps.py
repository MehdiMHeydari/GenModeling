"""
Compare CD exp_3 vs exp_4 vs exp_5: sample from the latest checkpoint of each
and plot a side-by-side grid showing the effect of batch size.

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

    exps = {
        "exp_3": {"dir": "darcy_student/exp_3", "label": "grad_accum=1\n(eff. batch 64)"},
        "exp_4": {"dir": "darcy_student/exp_4", "label": "grad_accum=4\n(eff. batch 256)"},
        "exp_5": {"dir": "darcy_student/exp_5", "label": "grad_accum=8\n(eff. batch 512)"},
    }

    # Find latest checkpoints
    for name, info in exps.items():
        ckpt = find_latest_checkpoint(info["dir"])
        assert ckpt, f"No checkpoints found for {name}"
        epoch = re.search(r"checkpoint_(\d+)", ckpt).group(1)
        info["ckpt"] = ckpt
        info["epoch"] = epoch
        print(f"{name}: {ckpt} (epoch {epoch})")

    # Sample from all
    all_samples = {}
    for name, info in exps.items():
        print(f"Sampling from {name}...")
        samples = load_and_sample(info["ckpt"], n, device)
        all_samples[name] = denormalize(samples, stats_dir)

    # Plot side-by-side grid
    n_exps = len(exps)
    fig, axes = plt.subplots(n_exps, n, figsize=(2.5 * n, 2.8 * n_exps))

    for i, (name, info) in enumerate(exps.items()):
        for j in range(n):
            img = all_samples[name][j, 0].numpy()
            axes[i, j].imshow(img, cmap="RdBu_r")
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(f"{name} (ep {info['epoch']})\n{info['label']}", fontsize=11)

    fig.suptitle("CD: Effect of Batch Size (exp_3 vs exp_4 vs exp_5)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison to {args.save_path}")


if __name__ == "__main__":
    main()
