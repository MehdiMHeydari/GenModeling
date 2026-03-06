"""
Generate samples from a trained Rectified Flow model.

Supports variable number of Euler steps to evaluate few-step quality.

Usage:
    python scripts/sample_rectified_flow.py \
        --checkpoint darcy_rectified_flow/exp_1/saved_state/checkpoint_799.pt \
        --norm_dir darcy_rectified_flow/exp_1/saved_state \
        --num_steps 100 --n_samples 1000 --save_path rf_samples.pt

    # Quick visual check
    python scripts/sample_rectified_flow.py \
        --checkpoint darcy_rectified_flow/exp_1/saved_state/checkpoint_799.pt \
        --norm_dir darcy_rectified_flow/exp_1/saved_state \
        --num_steps 1 --n_samples 8 --plot
"""

import argparse
import os
import torch as th
import numpy as np

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.flow_models import RectifiedFlowMatching
from src.inference.samplers import RectifiedFlowSampler


DATA_SHAPE = (1, 128, 128)

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


def load_norm_stats(norm_dir):
    min_path = os.path.join(norm_dir, "data_min.npy")
    max_path = os.path.join(norm_dir, "data_max.npy")
    if os.path.exists(min_path) and os.path.exists(max_path):
        return float(np.load(min_path)), float(np.load(max_path))
    return None, None


def denormalize(samples, data_min, data_max):
    if data_min is not None and data_max is not None:
        return (samples + 1.0) / 2.0 * (data_max - data_min) + data_min
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--norm_dir", type=str, default=None,
                        help="Dir with data_min.npy/data_max.npy for denormalization")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of Euler steps (1=one-step, 100=high quality)")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_path", type=str, default="rf_samples.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--plot", action="store_true",
                        help="Plot first 8 samples instead of saving all")
    args = parser.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    th.manual_seed(args.seed)

    # --- Build model ---
    network = UNetModel(**UNET_CFG)
    model = RectifiedFlowMatching(network=network, add_heavy_noise=False, infer=True)

    state = th.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- Sample ---
    sampler = RectifiedFlowSampler(model)
    n_total = args.n_samples
    batch_size = args.batch_size
    rounds = (n_total + batch_size - 1) // batch_size
    generated = []

    print(f"Generating {n_total} samples with {args.num_steps} Euler step(s)...")

    for i in range(rounds):
        n = min(batch_size, n_total - i * batch_size)
        z = th.randn(n, *DATA_SHAPE, device=device)
        samples = sampler.sample(z, num_steps=args.num_steps)
        generated.append(samples.cpu())
        if (i + 1) % 10 == 0 or i == rounds - 1:
            print(f"  Batch {i+1}/{rounds}")

    generated = th.cat(generated, dim=0)[:n_total]

    # --- Denormalize ---
    norm_dir = args.norm_dir
    if norm_dir is None:
        # Try to infer from checkpoint path
        norm_dir = os.path.dirname(args.checkpoint)
    data_min, data_max = load_norm_stats(norm_dir)
    generated_denorm = denormalize(generated, data_min, data_max)

    if data_min is not None:
        print(f"Denormalized range: [{generated_denorm.min():.4f}, {generated_denorm.max():.4f}]")

    # --- Output ---
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_show = min(8, n_total)
        fig, axes = plt.subplots(1, n_show, figsize=(2.5 * n_show, 3))
        if n_show == 1:
            axes = [axes]
        for j in range(n_show):
            axes[j].imshow(generated_denorm[j, 0].numpy(), cmap="RdBu_r")
            axes[j].axis("off")
            axes[j].set_title(f"Sample {j+1}", fontsize=9)
        fig.suptitle(f"Rectified Flow ({args.num_steps} steps)", fontweight="bold")
        plt.tight_layout()
        plot_path = args.save_path.replace(".pt", ".png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
    else:
        th.save(generated, args.save_path)
        print(f"Saved {n_total} samples to {args.save_path}")


if __name__ == "__main__":
    main()
