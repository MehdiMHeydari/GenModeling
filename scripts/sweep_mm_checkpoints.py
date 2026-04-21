"""
Sample each MM experiment at multiple training epochs to find where
(and how) mode collapse sets in. Produces one big grid figure:

    rows = one per (experiment, epoch)
    cols = N_SHOW samples from fixed seed
    row label includes the diversity metric (mean pairwise L2 between samples)

Fast version: the UNet and consistency model are built ONCE, and state_dicts
are swapped between checkpoints. Typical sweep of ~12 checkpoints runs in
~30 seconds on an A100.

Usage:
    python scripts/sweep_mm_checkpoints.py --gpu 0
    python scripts/sweep_mm_checkpoints.py --gpu 0 --epochs 25 50 75 100
"""

import argparse
import os

import numpy as np
import h5py
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.constants import (
    DATA_PATH, DATA_SHAPE, SCHEDULE_S, STATS_DIR, SINGLE_SEED,
    load_test_indices,
)
from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.consistency_models import MultistepConsistencyModel
from src.inference.samplers import MultistepCMSampler


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

# (experiment_dir, short_label, max_epoch)
MM_EXPERIMENTS = [
    ("darcy_student/exp_20", "exp_20 (mu=4, var=150)", 750),
    ("darcy_student/exp_21", "exp_21 (mu=4, var=200)", 999),
    ("darcy_student/exp_22", "exp_22 (mu=16, var=150)", 999),
]


def denormalize(samples, data_min, data_max):
    if isinstance(samples, th.Tensor):
        samples = samples.cpu().numpy()
    return (samples + 1.0) / 2.0 * (data_max - data_min) + data_min


def diversity_score(samples):
    flat = samples.reshape(samples.shape[0], -1)
    n = flat.shape[0]
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(flat[i] - flat[j]))
            count += 1
    return total / max(count, 1)


@th.no_grad()
def sample_from_state(model, sampler, state_path, noise, device):
    state = th.load(state_path, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    if "ema_state_dict" in state and model.ema_network is not None:
        model.ema_network.load_state_dict(state["ema_state_dict"])
    model.to(device).eval()
    return sampler.sample(noise.to(device)).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, nargs="+",
                   default=[100, 250, 500, 750, 999])
    p.add_argument("--n_show", type=int, default=8)
    p.add_argument("--student_steps", type=int, default=16)
    p.add_argument("--output", type=str,
                   default="results/figures/mm_checkpoint_sweep.png")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if th.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Build model + sampler ONCE
    network = UNetModel(**UNET_CFG)
    model = MultistepConsistencyModel(
        network=network, student_steps=args.student_steps,
        schedule_s=SCHEDULE_S, infer=True,
    )
    model.to(device).eval()
    sampler = MultistepCMSampler(model)

    # GT reference
    data_min = np.load(os.path.join(STATS_DIR, "data_min.npy"))
    data_max = np.load(os.path.join(STATS_DIR, "data_max.npy"))
    with h5py.File(DATA_PATH, "r") as f:
        data = f["tensor"][:]
    if data.ndim == 3:
        data = data[:, None]
    test_idx = load_test_indices()
    gt_denorm = data[test_idx[:args.n_show]]
    gt_diversity = diversity_score(gt_denorm)

    # Deterministic noise
    gen = th.Generator().manual_seed(SINGLE_SEED)
    noise = th.randn(args.n_show, *DATA_SHAPE, generator=gen)

    rows = [("Ground Truth", gt_denorm, gt_diversity)]

    for exp_dir, label, max_epoch in MM_EXPERIMENTS:
        for epoch in args.epochs:
            if epoch > max_epoch:
                continue
            ckpt = os.path.join(exp_dir, "saved_state", f"checkpoint_{epoch}.pt")
            if not os.path.exists(ckpt):
                print(f"[skip] {ckpt} missing")
                continue
            print(f"Sampling {label} epoch={epoch}...")
            samples = sample_from_state(model, sampler, ckpt, noise, device)
            denorm = denormalize(samples, data_min, data_max)
            div = diversity_score(denorm)
            rows.append((f"{label}\nepoch={epoch}  div={div:.3f}", denorm, div))

    # Plot
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, args.n_show,
        figsize=(1.6 * args.n_show, 1.6 * n_rows),
        squeeze=False,
    )
    for i, (label, arr, _) in enumerate(rows):
        for j in range(args.n_show):
            axes[i, j].imshow(arr[j, 0], cmap=CMAP)
            axes[i, j].axis("off")
        axes[i, 0].text(
            -0.1, 0.5, label,
            transform=axes[i, 0].transAxes,
            ha="right", va="center", fontsize=9,
            fontweight="bold" if i == 0 else "normal",
        )

    fig.tight_layout()
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved {args.output}")
    print(f"\nDiversity scores (higher = more diverse; GT = {gt_diversity:.3f}):")
    for label, _, div in rows:
        short = label.replace("\n", " ")
        print(f"  {short:60s}  div={div:.3f}  ratio={div/gt_diversity:.2f}")


if __name__ == "__main__":
    main()
