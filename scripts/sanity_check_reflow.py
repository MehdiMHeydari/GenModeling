"""
Sanity-check whether NS Reflow @ 1 step is actually doing what the WD
metric suggests, or if we're being fooled by some artifact.

Runs four tests (each prints/plots its own result):

  1. Memorization (nearest-neighbor):
        For each generated sample, distance to the closest training
        sample. Compared against the same metric on held-out GT samples.
        If Reflow's NN-distances are much smaller than held-out's,
        the model is regurgitating training data.

  2. Physics (incompressibility):
        Real NS is approximately incompressible: |div(v)| ~= 0 pointwise.
        Compute mean |div(v)| for Reflow samples and GT samples; if
        Reflow violates incompressibility but GT doesn't, the marginals
        match but the field isn't real fluid.

  3. Fresh-GT robustness:
        Recompute WD using random samples from the full NS dataset
        (not just locked test indices). If WD stays around 0.010, the
        result is independent of which test slice was used.

  4. Noise interpolation:
        Sample z_t = (1-t)*z1 + t*z2 for t in linspace(0,1,11), generate
        x_t through Reflow @ 1 step, plot the strip. Smooth path =
        Reflow learned a continuous mapping; discontinuous jumps =
        Reflow has discrete memorized modes.

Usage:
    python scripts/sanity_check_reflow.py --gpu 6
"""

import argparse
import os

import h5py
import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

from src.eval.constants import get_dataset, load_test_indices
from scripts.evaluate_paper import (
    denormalize, make_noise, make_unet_cfg, sample_rf, sample_teacher,
    to_scalar_field,
)


N_SAMPLES = 100
SEED = 0
TRAIN_NN_SUBSET = 2000   # NN search uses this many training frames
REFLOW_CKPT = "ns_rectified_flow_reflow/exp_1/saved_state/checkpoint_200.pt"
TEACHER_CKPT = "ns_teacher/exp_1/saved_state/checkpoint_75.pt"
OUT_DIR = "diagnostics/reflow_sanity"


def divergence(vfield):
    """vfield: (N, 2, H, W) numpy array of (Vx, Vy). Returns (N, H, W) of div v."""
    vx = vfield[:, 0]
    vy = vfield[:, 1]
    dvx_dx = np.gradient(vx, axis=2)
    dvy_dy = np.gradient(vy, axis=1)
    return dvx_dx + dvy_dy


def nn_to_training(samples_flat, train_flat, batch_size=50):
    """For each row in samples_flat, find min L2 distance to any row in
    train_flat. Done in batches to stay memory-friendly."""
    out = np.empty(samples_flat.shape[0], dtype=np.float32)
    train_norms2 = (train_flat ** 2).sum(axis=1)
    for i in range(0, samples_flat.shape[0], batch_size):
        batch = samples_flat[i:i + batch_size]
        batch_norms2 = (batch ** 2).sum(axis=1, keepdims=True)
        dot = batch @ train_flat.T
        d2 = batch_norms2 + train_norms2[None, :] - 2 * dot
        d2 = np.maximum(d2, 0.0)
        out[i:i + batch_size] = np.sqrt(d2.min(axis=1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if th.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = get_dataset("ns")
    data_shape = ds["data_shape"]
    unet_cfg = make_unet_cfg(data_shape)
    data_min = np.load(os.path.join(ds["stats_dir"], "data_min.npy"))
    data_max = np.load(os.path.join(ds["stats_dir"], "data_max.npy"))

    # --- Load all GT data once ---
    with h5py.File(ds["data_path"], "r") as f:
        data = f["tensor"][:]
    if data.ndim == 3:
        data = data[:, None]
    train_n = ds["train_samples"]

    test_idx = load_test_indices("ns")
    held_out = data[test_idx[:N_SAMPLES]]

    fresh_rng = np.random.RandomState(SEED + 7)
    fresh_idx = fresh_rng.choice(len(data), size=N_SAMPLES, replace=False)
    fresh_gt = data[fresh_idx]

    # NN target = subset of training set
    nn_rng = np.random.RandomState(0)
    train_pool_idx = nn_rng.choice(train_n, size=TRAIN_NN_SUBSET, replace=False)
    train_pool = data[train_pool_idx]
    train_flat = train_pool.reshape(TRAIN_NN_SUBSET, -1).astype(np.float32)

    # --- Sample Reflow and Teacher ---
    noise = make_noise(SEED, N_SAMPLES, data_shape)

    print(f"Sampling Reflow @ 1 step ({N_SAMPLES} frames)...")
    reflow_samples, _ = sample_rf(REFLOW_CKPT, 1, noise, device, unet_cfg, batch_size=64)
    reflow_denorm = denormalize(reflow_samples, data_min, data_max)

    print(f"Sampling Teacher @ 75 step ({N_SAMPLES} frames)...")
    teacher_samples, _ = sample_teacher(TEACHER_CKPT, 75, noise, device, unet_cfg, batch_size=32)
    teacher_denorm = denormalize(teacher_samples, data_min, data_max)

    # ---------------------------------------------------------------
    # Test 1: nearest-neighbor distance to training set
    # ---------------------------------------------------------------
    print("\n[1/4] Memorization (NN to training set)")
    reflow_flat = reflow_denorm.reshape(N_SAMPLES, -1).astype(np.float32)
    teacher_flat = teacher_denorm.reshape(N_SAMPLES, -1).astype(np.float32)
    held_out_flat = held_out.reshape(N_SAMPLES, -1).astype(np.float32)
    fresh_flat = fresh_gt.reshape(N_SAMPLES, -1).astype(np.float32)

    nn_reflow = nn_to_training(reflow_flat, train_flat)
    nn_teacher = nn_to_training(teacher_flat, train_flat)
    nn_held_out = nn_to_training(held_out_flat, train_flat)
    nn_fresh = nn_to_training(fresh_flat, train_flat)

    print(f"  NN-to-train  Reflow @ 1:      mean={nn_reflow.mean():.3f}  std={nn_reflow.std():.3f}")
    print(f"  NN-to-train  Teacher @ 75:    mean={nn_teacher.mean():.3f}  std={nn_teacher.std():.3f}")
    print(f"  NN-to-train  held-out test:   mean={nn_held_out.mean():.3f}  std={nn_held_out.std():.3f}")
    print(f"  NN-to-train  fresh GT (rand): mean={nn_fresh.mean():.3f}  std={nn_fresh.std():.3f}")
    print("  → if Reflow << held-out, suspicious memorization.")
    print("  → if Reflow ~= held-out, not memorizing.")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(
        min(nn_reflow.min(), nn_held_out.min(), nn_teacher.min(), nn_fresh.min()),
        max(nn_reflow.max(), nn_held_out.max(), nn_teacher.max(), nn_fresh.max()),
        40)
    ax.hist(nn_held_out, bins=bins, density=True, alpha=0.35, label="held-out GT", color="gray")
    ax.hist(nn_fresh, bins=bins, density=True, histtype="step", linewidth=1.8,
            label="fresh GT (random)", color="black")
    ax.hist(nn_teacher, bins=bins, density=True, histtype="step", linewidth=1.8,
            label="Teacher @ 75", color="C1")
    ax.hist(nn_reflow, bins=bins, density=True, histtype="step", linewidth=2.0,
            label="Reflow @ 1", color="C0")
    ax.set_xlabel("L2 distance to nearest training sample")
    ax.set_ylabel("Density")
    ax.set_title(f"NN distance to training (subset of {TRAIN_NN_SUBSET})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "nn_distances.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------
    # Test 2: physics (incompressibility)
    # ---------------------------------------------------------------
    print("\n[2/4] Physics (mean |div v|)")
    div_reflow = np.abs(divergence(reflow_denorm)).mean(axis=(1, 2))
    div_teacher = np.abs(divergence(teacher_denorm)).mean(axis=(1, 2))
    div_held_out = np.abs(divergence(held_out)).mean(axis=(1, 2))
    div_fresh = np.abs(divergence(fresh_gt)).mean(axis=(1, 2))

    print(f"  mean |div v|  Reflow @ 1:      mean={div_reflow.mean():.5f}  std={div_reflow.std():.5f}")
    print(f"  mean |div v|  Teacher @ 75:    mean={div_teacher.mean():.5f}  std={div_teacher.std():.5f}")
    print(f"  mean |div v|  held-out test:   mean={div_held_out.mean():.5f}  std={div_held_out.std():.5f}")
    print(f"  mean |div v|  fresh GT (rand): mean={div_fresh.mean():.5f}  std={div_fresh.std():.5f}")
    print("  → Reflow much higher than GT means it's producing non-physical fields.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([div_held_out, div_fresh, div_teacher, div_reflow],
               tick_labels=["held-out GT", "fresh GT", "Teacher @ 75", "Reflow @ 1"])
    ax.set_ylabel("mean |div v| per sample")
    ax.set_title("Incompressibility check")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "divergence.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------
    # Test 3: WD against fresh GT, not just locked test indices
    # ---------------------------------------------------------------
    print("\n[3/4] WD robustness (vs fresh random GT)")
    sub_rng = np.random.RandomState(0)
    cap = 100_000
    reflow_field = to_scalar_field(reflow_denorm).flatten()
    held_out_field = to_scalar_field(held_out).flatten()
    fresh_field = to_scalar_field(fresh_gt).flatten()
    teacher_field = to_scalar_field(teacher_denorm).flatten()

    def wd(a, b):
        a_idx = sub_rng.choice(len(a), size=min(cap, len(a)), replace=False)
        b_idx = sub_rng.choice(len(b), size=min(cap, len(b)), replace=False)
        return float(wasserstein_distance(a[a_idx], b[b_idx]))

    wd_reflow_held = wd(reflow_field, held_out_field)
    wd_reflow_fresh = wd(reflow_field, fresh_field)
    wd_teacher_held = wd(teacher_field, held_out_field)
    wd_teacher_fresh = wd(teacher_field, fresh_field)
    print(f"  Reflow vs held-out:    WD={wd_reflow_held:.4f}")
    print(f"  Reflow vs fresh GT:    WD={wd_reflow_fresh:.4f}")
    print(f"  Teacher vs held-out:   WD={wd_teacher_held:.4f}")
    print(f"  Teacher vs fresh GT:   WD={wd_teacher_fresh:.4f}")
    print("  → Reflow's WD should be ~same on held-out and fresh GT.")

    # ---------------------------------------------------------------
    # Test 4: noise interpolation (linear + spherical)
    # ---------------------------------------------------------------
    # Linear interpolation z_t = (1-t)z1 + t*z2 reduces |z_t| at t=0.5
    # to ~sqrt(0.5)*|z|, putting the model OOD relative to the unit-variance
    # noise it was trained on. SLERP keeps |z_t| constant — proper test of
    # whether the model has a continuous mapping in noise space.
    print("\n[4/4] Noise interpolation through Reflow (linear + SLERP)")
    n_steps = 11
    th.manual_seed(SEED)
    z1 = th.randn(1, *data_shape)
    z2 = th.randn(1, *data_shape)
    ts = np.linspace(0, 1, n_steps)

    # --- Linear ---
    z_lin = th.cat([(1 - t) * z1 + t * z2 for t in ts], dim=0)
    samples_lin, _ = sample_rf(REFLOW_CKPT, 1, z_lin, device, unet_cfg,
                               batch_size=n_steps)
    mag_lin = to_scalar_field(denormalize(samples_lin, data_min, data_max))

    # --- Spherical (SLERP) ---
    # theta = angle between z1 and z2 viewed as unit vectors in flat space
    z1_flat = z1.flatten()
    z2_flat = z2.flatten()
    cos_theta = float((z1_flat @ z2_flat) /
                      (z1_flat.norm() * z2_flat.norm() + 1e-12))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = float(np.arccos(cos_theta))
    sin_theta = float(np.sin(theta)) if theta > 1e-6 else 1.0

    z_slerp_list = []
    for t in ts:
        if theta < 1e-6:
            zt = (1 - t) * z1 + t * z2
        else:
            a = float(np.sin((1 - t) * theta) / sin_theta)
            b = float(np.sin(t * theta) / sin_theta)
            zt = a * z1 + b * z2
        z_slerp_list.append(zt)
    z_slerp = th.cat(z_slerp_list, dim=0)
    samples_slerp, _ = sample_rf(REFLOW_CKPT, 1, z_slerp, device, unet_cfg,
                                 batch_size=n_steps)
    mag_slerp = to_scalar_field(denormalize(samples_slerp, data_min, data_max))

    fig, axes = plt.subplots(2, n_steps, figsize=(2.0 * n_steps, 4.8))
    for i in range(n_steps):
        axes[0, i].imshow(mag_lin[i])
        axes[0, i].axis("off")
        axes[0, i].set_title(f"t={ts[i]:.2f}", fontsize=8)
        axes[1, i].imshow(mag_slerp[i])
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Linear", fontsize=11, rotation=0,
                          labelpad=40, ha="right", va="center")
    axes[1, 0].set_ylabel("SLERP", fontsize=11, rotation=0,
                          labelpad=40, ha="right", va="center")
    for i in [0, 1]:
        axes[i, 0].axis("on")
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        for spine in axes[i, 0].spines.values():
            spine.set_visible(False)
    fig.suptitle("Reflow @ 1 step: noise interpolation (linear vs spherical)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "noise_interpolation.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Linear (top row): goes OOD at intermediate t — variance drops")
    print("  SLERP  (bot row): preserves |z|=const, true test of continuity")
    print("  → SLERP smooth = continuous mapping. SLERP jumps = memorized modes.")

    print(f"\nDone. Plots in {OUT_DIR}/.")


if __name__ == "__main__":
    main()
