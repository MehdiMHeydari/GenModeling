"""
Generate a grid comparison of RF and Reflow samples across different step counts.

Produces one large figure:
  - Two column groups: "Rectified Flow" and "Reflow"
  - Each row = a different step count (1, 2, 3, ..., 10)
  - Each row shows N sample images side by side
  - Uses the SAME initial noise across all step counts and both models

Usage:
    python scripts/rf_step_sweep.py \
        --rf_checkpoint darcy_rectified_flow/exp_1/saved_state/checkpoint_799.pt \
        --reflow_checkpoint darcy_rectified_flow_reflow/exp_1/saved_state/checkpoint_399.pt \
        --norm_dir darcy_rectified_flow/exp_1/saved_state \
        --gpu 5
"""

import argparse
import os
import torch as th
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

STEP_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_SHOW = 6  # samples per row


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


def load_model(checkpoint_path, device):
    network = UNetModel(**UNET_CFG)
    model = RectifiedFlowMatching(network=network, add_heavy_noise=False, infer=True)
    state = th.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model


def sample_all_steps(model, initial_noise, step_counts, device):
    """Sample with each step count using the same initial noise."""
    sampler = RectifiedFlowSampler(model)
    results = {}
    for n_steps in step_counts:
        z = initial_noise.clone().to(device)
        with th.no_grad():
            samples = sampler.sample(z, num_steps=n_steps)
        results[n_steps] = samples.cpu()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf_checkpoint", type=str, required=True,
                        help="Path to RF round-1 checkpoint")
    parser.add_argument("--reflow_checkpoint", type=str, default=None,
                        help="Path to reflow checkpoint (omit to show RF only)")
    parser.add_argument("--norm_dir", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="rf_step_sweep.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_show", type=int, default=N_SHOW,
                        help="Number of samples per row")
    args = parser.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    th.manual_seed(args.seed)

    # Fixed initial noise — same for all steps and both models
    initial_noise = th.randn(args.n_show, *DATA_SHAPE)

    # Norm stats
    norm_dir = args.norm_dir or os.path.dirname(args.rf_checkpoint)
    data_min, data_max = load_norm_stats(norm_dir)

    # --- Load and sample RF ---
    print("Loading RF model...")
    rf_model = load_model(args.rf_checkpoint, device)
    print("Sampling RF across step counts...")
    rf_results = sample_all_steps(rf_model, initial_noise, STEP_COUNTS, device)
    del rf_model
    th.cuda.empty_cache()

    # --- Load and sample Reflow (if provided) ---
    has_reflow = args.reflow_checkpoint is not None
    if has_reflow:
        print("Loading Reflow model...")
        reflow_model = load_model(args.reflow_checkpoint, device)
        print("Sampling Reflow across step counts...")
        reflow_results = sample_all_steps(reflow_model, initial_noise, STEP_COUNTS, device)
        del reflow_model
        th.cuda.empty_cache()

    # --- Build figure ---
    n_rows = len(STEP_COUNTS)
    n_cols = args.n_show
    n_groups = 2 if has_reflow else 1
    total_cols = n_cols * n_groups

    # Add gap between RF and Reflow columns
    fig_width = 2.0 * total_cols + (1.5 if has_reflow else 0) + 1.5
    fig_height = 2.0 * n_rows + 1.5

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Use gridspec for gap between the two groups
    if has_reflow:
        gs = fig.add_gridspec(n_rows, total_cols + 1,  # +1 for spacer column
                              wspace=0.05, hspace=0.15,
                              left=0.06, right=0.98, top=0.93, bottom=0.02,
                              width_ratios=[1]*n_cols + [0.3] + [1]*n_cols)
    else:
        gs = fig.add_gridspec(n_rows, n_cols,
                              wspace=0.05, hspace=0.15,
                              left=0.06, right=0.98, top=0.93, bottom=0.02)

    for row_idx, n_steps in enumerate(STEP_COUNTS):
        # RF samples
        rf_denorm = denormalize(rf_results[n_steps], data_min, data_max)
        for col_idx in range(args.n_show):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(rf_denorm[col_idx, 0].numpy(), cmap="RdBu_r")
            ax.axis("off")
            # Row label on leftmost column
            if col_idx == 0:
                ax.text(-0.15, 0.5, f"{n_steps} step{'s' if n_steps > 1 else ''}",
                        transform=ax.transAxes, fontsize=11, fontweight="bold",
                        va="center", ha="right")

        # Reflow samples
        if has_reflow:
            reflow_denorm = denormalize(reflow_results[n_steps], data_min, data_max)
            for col_idx in range(args.n_show):
                # +1 for spacer column
                ax = fig.add_subplot(gs[row_idx, n_cols + 1 + col_idx])
                ax.imshow(reflow_denorm[col_idx, 0].numpy(), cmap="RdBu_r")
                ax.axis("off")

    # Column group titles
    rf_center = n_cols / 2 / total_cols * 0.92 + 0.06
    fig.text(rf_center, 0.97, "Rectified Flow (Round 1)",
             fontsize=14, fontweight="bold", ha="center")
    if has_reflow:
        reflow_center = (n_cols + 0.5 + n_cols / 2) / (total_cols + 1) * 0.92 + 0.06
        fig.text(reflow_center, 0.97, "Reflow (Round 2)",
                 fontsize=14, fontweight="bold", ha="center")

    plt.savefig(args.save_path, dpi=150, bbox_inches="tight")
    print(f"Saved step sweep plot to {args.save_path}")


if __name__ == "__main__":
    main()
