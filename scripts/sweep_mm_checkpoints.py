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
    TEACHER_CKPT, TEACHER_DDIM_STEPS, load_test_indices,
)
from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.consistency_models import MultistepConsistencyModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.diffusion_utils import ddim_step
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
    """Mean pairwise L2 distance over flattened pixels.
    Captures *pixel-level* variation (including same-shape-different-size)."""
    flat = samples.reshape(samples.shape[0], -1)
    n = flat.shape[0]
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(flat[i] - flat[j]))
            count += 1
    return total / max(count, 1)


def structural_diversity(samples):
    """
    Mean pairwise L2 distance of per-sample center-of-mass.

    Captures *structural* variation: where is the mass in each field?
    - High value: each sample puts its active region in a different spatial location.
    - Low value: all samples put their mass in the same place (mode collapse in shape).

    Complements `diversity_score` which can be fooled by same-shape-different-size.
    """
    n, c, h, w = samples.shape
    y_coords, x_coords = np.meshgrid(
        np.arange(h), np.arange(w), indexing="ij"
    )
    coms = np.zeros((n, 2))
    for i in range(n):
        arr = samples[i, 0]
        shifted = arr - arr.min() + 1e-8  # shift to positive for weighting
        total = shifted.sum()
        coms[i, 0] = (x_coords * shifted).sum() / total
        coms[i, 1] = (y_coords * shifted).sum() / total

    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += float(np.linalg.norm(coms[i] - coms[j]))
            count += 1
    return total_dist / max(count, 1)


@th.no_grad()
def sample_from_state(model, sampler, state_path, noise, device):
    state = th.load(state_path, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    if "ema_state_dict" in state and model.ema_network is not None:
        model.ema_network.load_state_dict(state["ema_state_dict"])
    model.to(device).eval()
    return sampler.sample(noise.to(device)).cpu()


@th.no_grad()
def sample_teacher(noise, device, n_steps=TEACHER_DDIM_STEPS):
    network = UNetModel(**UNET_CFG)
    teacher = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(TEACHER_CKPT, map_location="cpu", weights_only=True)
    teacher.network.load_state_dict(state["model_state_dict"])
    teacher.to(device).eval()

    ts = th.linspace(1.0, 0.0, n_steps + 1, device=device)
    z = noise.to(device)
    n = z.shape[0]
    for step in range(n_steps):
        t = th.full((n,), ts[step].item(), device=device)
        s = th.full((n,), ts[step + 1].item(), device=device)
        x_hat = teacher.predict_x(z, t)
        z = ddim_step(x_hat, z, t, s, SCHEDULE_S)
    samples = z.cpu()
    del teacher, network
    th.cuda.empty_cache()
    return samples


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
    gt_struct = structural_diversity(gt_denorm)

    # Deterministic noise
    gen = th.Generator().manual_seed(SINGLE_SEED)
    noise = th.randn(args.n_show, *DATA_SHAPE, generator=gen)

    def make_row(label, denorm):
        d = diversity_score(denorm)
        s = structural_diversity(denorm)
        label_with_metrics = f"{label}\npix={d:.2f}  struct={s:.2f}"
        return (label_with_metrics, denorm, d, s)

    rows = [(f"Ground Truth\npix={gt_diversity:.2f}  struct={gt_struct:.2f}",
             gt_denorm, gt_diversity, gt_struct)]

    # Teacher reference — same noise as MM students get
    print("Sampling Teacher (reference target)...")
    teacher_samples = sample_teacher(noise, device)
    teacher_denorm = denormalize(teacher_samples, data_min, data_max)
    rows.append(make_row("Teacher (ref target)", teacher_denorm))

    # CD baseline (no moment matching) for reference
    cd_baseline_ckpt = "darcy_student/exp_3/saved_state/checkpoint_999.pt"
    if os.path.exists(cd_baseline_ckpt):
        print("Sampling CD baseline (no moment)...")
        cd_samples = sample_from_state(model, sampler, cd_baseline_ckpt,
                                       noise, device)
        cd_denorm = denormalize(cd_samples, data_min, data_max)
        rows.append(make_row("CD baseline (no moment)", cd_denorm))

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
            rows.append(make_row(f"{label} epoch={epoch}", denorm))

    # Plot
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, args.n_show,
        figsize=(1.6 * args.n_show, 1.6 * n_rows),
        squeeze=False,
    )
    for i, (label, arr, _, _) in enumerate(rows):
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
    print(f"\nDiversity scores:")
    print(f"  pix = mean pairwise L2 (captures any pixel variation)")
    print(f"  struct = center-of-mass pairwise L2 (captures spatial location variation)")
    print(f"  GT reference: pix={gt_diversity:.2f}  struct={gt_struct:.2f}")
    print()
    print(f"  {'method':<40s}  {'pix':>7s}  {'struct':>7s}  "
          f"{'pix/GT':>7s}  {'struct/GT':>9s}")
    for label, _, div, struct in rows:
        short = label.replace("\n", " ")
        print(f"  {short:<40s}  {div:7.2f}  {struct:7.2f}  "
              f"{div/gt_diversity:7.2f}  {struct/gt_struct:9.2f}")


if __name__ == "__main__":
    main()
