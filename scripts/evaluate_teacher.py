"""
Evaluate a teacher checkpoint by generating samples and comparing to real data.
Can be run while training is still in progress.

Usage:
    python scripts/evaluate_teacher.py darcy_teacher/exp_1/saved_state/checkpoint_25.pt
    python scripts/evaluate_teacher.py darcy_teacher/exp_1/saved_state/checkpoint_25.pt --ddim  # deterministic DDIM
"""

import sys
import os
import numpy as np
import h5py
import torch as th
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.diffusion_utils import addim_step, ddim_step, snr, alpha_t, sigma_t

# ============================================================
# CONFIG
# ============================================================
DATA_SHAPE = (1, 128, 128)
DATA_PATH = "data/2D_DarcyFlow_beta1.0_Train.hdf5"
SCHEDULE_S = 0.008
DEVICE = "cuda"

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

NUM_SAMPLES = 16
BATCH_SIZE = 8
DDIM_STEPS = 50


def main():
    ckpt_path = sys.argv[1]
    use_ddim = "--ddim" in sys.argv

    epoch = ckpt_path.split("checkpoint_")[1].split(".")[0]
    sampler_name = "DDIM" if use_ddim else "aDDIM"
    output_dir = f"eval_teacher_ep{epoch}"
    os.makedirs(output_dir, exist_ok=True)

    # --- Load real data ---
    with h5py.File(DATA_PATH, 'r') as f:
        outputs = np.array(f['tensor']).astype(np.float32)
    if outputs.ndim == 3:
        outputs = outputs[:, np.newaxis, :, :]

    # Find norm stats
    save_dir = os.path.dirname(ckpt_path)
    if not os.path.exists(os.path.join(save_dir, "data_min.npy")):
        # Check other common locations
        for d in ["darcy_teacher/exp_1/saved_state", "darcy_teacher_v1/exp_1/saved_state",
                   "darcy_student/exp_1/saved_state"]:
            if os.path.exists(os.path.join(d, "data_min.npy")):
                save_dir = d
                break

    data_min = float(np.load(os.path.join(save_dir, "data_min.npy")))
    data_max = float(np.load(os.path.join(save_dir, "data_max.npy")))
    print(f"Norm stats from {save_dir}: [{data_min:.4f}, {data_max:.4f}]")

    def denormalize(x_norm):
        if isinstance(x_norm, th.Tensor):
            x_norm = x_norm.numpy()
        return (x_norm + 1.0) / 2.0 * (data_max - data_min) + data_min

    real_norm = 2.0 * (outputs - data_min) / (data_max - data_min) - 1.0
    test_data = real_norm[9500:]

    # --- Load teacher ---
    print(f"\nLoading checkpoint: {ckpt_path}")
    network = UNetModel(**UNET_CFG)
    teacher = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(ckpt_path, map_location='cpu', weights_only=True)

    if 'ema_state_dict' in state:
        teacher.network.load_state_dict(state['ema_state_dict'])
        print("Using EMA weights")
    else:
        teacher.network.load_state_dict(state['model_state_dict'])
        print("Using raw weights (no EMA found)")
    teacher.to(DEVICE)
    teacher.eval()
    print(f"Loaded epoch {state['epoch']}")

    # --- Generate samples ---
    print(f"\nGenerating {NUM_SAMPLES} samples ({DDIM_STEPS} {sampler_name} steps)...")
    C, H, W = DATA_SHAPE
    ts = th.linspace(1.0, 0.0, DDIM_STEPS + 1, device=DEVICE)
    rounds = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE

    th.manual_seed(42)
    all_samples = []

    with th.no_grad():
        for r in range(rounds):
            n = min(BATCH_SIZE, NUM_SAMPLES - r * BATCH_SIZE)
            z = th.randn(n, C, H, W, device=DEVICE)
            for i in range(DDIM_STEPS):
                t_batch = th.full((n,), ts[i].item(), device=DEVICE)
                s_batch = th.full((n,), ts[i + 1].item(), device=DEVICE)
                x_hat = teacher.predict_x(z, t_batch)
                if use_ddim:
                    z = ddim_step(x_hat, z, t_batch, s_batch, SCHEDULE_S)
                else:
                    x_var = 0.1 / (2.0 + snr(t_batch, SCHEDULE_S))
                    z = addim_step(x_hat, z, x_var, t_batch, s_batch, SCHEDULE_S)
            all_samples.append(z.cpu())

    samples = th.cat(all_samples, dim=0)[:NUM_SAMPLES]
    samples_denorm = denormalize(samples)
    real_denorm = denormalize(test_data)

    # --- Statistics ---
    print(f"\n{'='*50}")
    print(f"TEACHER EVALUATION (epoch {epoch}, {sampler_name})")
    print(f"{'='*50}")
    print(f"{'':20s} {'Teacher':>12s} {'Real (test)':>12s}")
    print(f"{'-'*50}")
    print(f"{'Mean':20s} {samples_denorm.mean():12.6f} {real_denorm.mean():12.6f}")
    print(f"{'Std':20s} {samples_denorm.std():12.6f} {real_denorm.std():12.6f}")
    print(f"{'Min':20s} {samples_denorm.min():12.6f} {real_denorm.min():12.6f}")
    print(f"{'Max':20s} {samples_denorm.max():12.6f} {real_denorm.max():12.6f}")

    s_means = samples_denorm.mean(axis=(1, 2, 3))
    r_means = real_denorm.mean(axis=(1, 2, 3))
    s_stds = samples_denorm.std(axis=(1, 2, 3))
    r_stds = real_denorm.std(axis=(1, 2, 3))

    print(f"\nPer-sample mean:  Teacher {s_means.mean():.4f} +/- {s_means.std():.4f}  |  Real {r_means.mean():.4f} +/- {r_means.std():.4f}")
    print(f"Per-sample std:   Teacher {s_stds.mean():.4f} +/- {s_stds.std():.4f}  |  Real {r_stds.mean():.4f} +/- {r_stds.std():.4f}")

    # --- Plot: samples grid + real comparison ---
    n_show = 4
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))

    vmin = min(samples_denorm[:n_show].min(), real_denorm[:n_show].min())
    vmax = max(samples_denorm[:n_show].max(), real_denorm[:n_show].max())

    for i in range(n_show):
        axes[0, i].imshow(samples_denorm[i, 0], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'Teacher {i+1}')
        axes[0, i].axis('off')

        im = axes[1, i].imshow(real_denorm[i, 0], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Real {i+1}')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel(f'Teacher ({sampler_name})', fontsize=12)
    axes[1, 0].set_ylabel('Real', fontsize=12)
    fig.colorbar(im, ax=axes, shrink=0.5, label='u(x,y)')
    plt.suptitle(f'Teacher Epoch {epoch} ({sampler_name}, {DDIM_STEPS} steps)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- Histogram ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(samples_denorm.flatten(), bins=100, alpha=0.5, density=True,
            label=f'Teacher ({sampler_name})', color='tab:blue')
    ax.hist(real_denorm.flatten(), bins=100, alpha=0.5, density=True,
            label='Real (test)', color='tab:orange')
    ax.set_xlabel('u(x,y)')
    ax.set_ylabel('Density')
    ax.set_title(f'Pixel Distribution: Teacher Epoch {epoch} vs Real')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/histogram.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


if __name__ == '__main__':
    main()
