"""
Diagnose Rectified Flow experiments: show training status and sample quality.

Checks both Round 1 and Reflow checkpoints, then generates a comparison
figure showing samples at different Euler step counts (1, 10, 100).

Usage:
    python scripts/diagnose_rectified_flow.py [--gpu 0] [--n_samples 8]
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

EXPERIMENTS = {
    "Round 1 (RF)": "darcy_rectified_flow/exp_1/saved_state",
    "Round 2 (Reflow)": "darcy_rectified_flow_reflow/exp_1/saved_state",
}

EULER_STEPS = [1, 10, 100]


def find_checkpoints(ckpt_dir):
    """Find all checkpoints and return sorted list of (epoch, path)."""
    pattern = os.path.join(ckpt_dir, "checkpoint_*.pt")
    ckpts = glob.glob(pattern)
    results = []
    for c in ckpts:
        m = re.search(r"checkpoint_(\d+)\.pt", c)
        if m:
            results.append((int(m.group(1)), c))
    return sorted(results)


def load_model(ckpt_path, device):
    """Load a rectified flow model from checkpoint."""
    network = UNetModel(**UNET_CFG)
    model = RectifiedFlowMatching(network=network, add_heavy_noise=False, infer=True)
    state = th.load(ckpt_path, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model


def denormalize(samples, norm_dir):
    """Denormalize from [-1, 1] to original range."""
    min_path = os.path.join(norm_dir, "data_min.npy")
    max_path = os.path.join(norm_dir, "data_max.npy")
    if os.path.exists(min_path) and os.path.exists(max_path):
        data_min = float(np.load(min_path))
        data_max = float(np.load(max_path))
        return (samples + 1.0) / 2.0 * (data_max - data_min) + data_min
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="rf_diagnosis.png")
    args = parser.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    n = args.n_samples
    norm_dir = "darcy_rectified_flow/exp_1/saved_state"

    # --- Step 1: Report status ---
    print("=" * 60)
    print(" Rectified Flow Experiment Status")
    print("=" * 60)

    available_exps = {}
    for name, ckpt_dir in EXPERIMENTS.items():
        ckpts = find_checkpoints(ckpt_dir)
        if ckpts:
            latest_epoch, latest_path = ckpts[-1]
            all_epochs = [e for e, _ in ckpts]
            print(f"\n{name}:")
            print(f"  Directory: {ckpt_dir}")
            print(f"  Checkpoints: {len(ckpts)} (epochs: {all_epochs})")
            print(f"  Latest: epoch {latest_epoch}")
            available_exps[name] = {"epoch": latest_epoch, "path": latest_path}
        else:
            print(f"\n{name}:")
            print(f"  Directory: {ckpt_dir}")
            print(f"  NO CHECKPOINTS FOUND")

    if not available_exps:
        print("\nNo checkpoints found. Nothing to sample.")
        return

    # --- Step 2: Sample from each experiment at different step counts ---
    print("\n" + "=" * 60)
    print(" Generating Samples")
    print("=" * 60)

    # Use same noise for fair comparison
    th.manual_seed(42)
    fixed_noise = th.randn(n, *DATA_SHAPE, device=device)

    # all_samples[exp_name][num_steps] = tensor
    all_samples = {}

    for name, info in available_exps.items():
        print(f"\nLoading {name} (epoch {info['epoch']})...")
        model = load_model(info["path"], device)
        sampler = RectifiedFlowSampler(model)
        all_samples[name] = {}

        for steps in EULER_STEPS:
            print(f"  Sampling with {steps} Euler step(s)...")
            with th.no_grad():
                samples = sampler.sample(fixed_noise.clone(), num_steps=steps)
            samples = denormalize(samples.cpu(), norm_dir)
            all_samples[name][steps] = samples

        del model
        th.cuda.empty_cache()

    # --- Step 3: Plot comparison ---
    print("\n" + "=" * 60)
    print(" Plotting")
    print("=" * 60)

    n_rows = len(available_exps) * len(EULER_STEPS)
    # Extra left margin for row labels
    fig, axes = plt.subplots(n_rows, n, figsize=(2.5 * n + 2, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    row = 0
    for name in available_exps:
        for steps in EULER_STEPS:
            samples = all_samples[name][steps]
            for j in range(n):
                img = samples[j, 0].numpy()
                axes[row, j].imshow(img, cmap="RdBu_r")
                axes[row, j].axis("off")
            epoch = available_exps[name]["epoch"]
            label = (
                f"{name}\n"
                f"epoch {epoch}\n"
                f"{steps} Euler step{'s' if steps > 1 else ''}"
            )
            axes[row, 0].set_ylabel(label, fontsize=11, fontweight="bold", labelpad=10)
            row += 1

    fig.suptitle(
        "Rectified Flow: Round 1 vs Reflow at Different Step Counts",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.subplots_adjust(left=0.12)
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.save_path}")


if __name__ == "__main__":
    main()
