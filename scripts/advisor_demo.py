"""
Generate two figures for advisor presentation:
  1. advisor_samples.png  — Sample grids (GT, Teacher, Students) + speedup table
  2. advisor_histograms.png — Pixel value distribution histograms

Usage:
    python scripts/advisor_demo.py
    python scripts/advisor_demo.py --device cuda --num-samples 64 --seed 42
"""

import os
import sys
import time
import argparse
import numpy as np
import h5py
import torch as th

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.consistency_models import MultistepConsistencyModel
from src.models.diffusion_utils import ddim_step


# ============================================================
# Config
# ============================================================
DATA_SHAPE = (1, 128, 128)
SCHEDULE_S = 0.008
DDIM_STEPS = 50

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

DATA_PATH = "data/2D_DarcyFlow_beta1.0_Train.hdf5"
TEACHER_CKPT = "darcy_teacher/exp_1/saved_state/checkpoint_200.pt"
NORM_DIR = "darcy_teacher/exp_1/saved_state"

STUDENT_CONFIGS = [
    {"name": "CD Student\n(4 steps)", "dir": "darcy_student/exp_1/saved_state", "steps": 4},
    {"name": "CD Student\n(8 steps)", "dir": "darcy_student/exp_2/saved_state", "steps": 8},
    {"name": "CD Student\n(16 steps)", "dir": "darcy_student/exp_3/saved_state", "steps": 16},
]


# ============================================================
# Helpers
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-samples", type=int, default=64,
                   help="Total samples to generate (grids show first 8)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="advisor_output")
    p.add_argument("--timing-runs", type=int, default=5)
    return p.parse_args()


def build_unet():
    return UNetModel(**UNET_CFG)


def find_latest_checkpoint(save_dir):
    if not os.path.isdir(save_dir):
        return None
    ckpts = [f for f in os.listdir(save_dir)
             if f.startswith('checkpoint_') and f.endswith('.pt')]
    if not ckpts:
        return None
    ckpts.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
    return os.path.join(save_dir, ckpts[-1])


def denormalize(x_norm, data_min, data_max):
    if isinstance(x_norm, th.Tensor):
        x_norm = x_norm.cpu().numpy()
    return (x_norm + 1.0) / 2.0 * (data_max - data_min) + data_min


# ============================================================
# Sampling (plain DDIM for teacher — verified working)
# ============================================================

def sample_teacher(teacher, z_batches, device):
    """Sample using plain DDIM (not aDDIM). Matches the verified working script."""
    ts = th.linspace(1.0, 0.0, DDIM_STEPS + 1, device=device)
    all_samples = []
    with th.no_grad():
        for z in z_batches:
            z = z.to(device)
            n = z.shape[0]
            for i in range(DDIM_STEPS):
                t_batch = th.full((n,), ts[i].item(), device=device)
                s_batch = th.full((n,), ts[i + 1].item(), device=device)
                x_hat = teacher.predict_x(z, t_batch)
                z = ddim_step(x_hat, z, t_batch, s_batch, SCHEDULE_S)
            all_samples.append(z.cpu())
    return th.cat(all_samples, dim=0)


def sample_student(model, z_batches, device):
    all_samples = []
    with th.no_grad():
        for z in z_batches:
            z = z.to(device)
            samples = model.sample(z)
            all_samples.append(samples.cpu())
    return th.cat(all_samples, dim=0)


# ============================================================
# Timing
# ============================================================

def time_model(sample_fn, z_batches, device, num_runs=5):
    """Returns seconds per sample."""
    timing_z = z_batches[:2]
    total = sum(z.shape[0] for z in timing_z)

    # Warmup
    sample_fn(timing_z, device)
    if device != 'cpu':
        th.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        if device != 'cpu':
            th.cuda.synchronize()
        t0 = time.perf_counter()
        sample_fn(timing_z, device)
        if device != 'cpu':
            th.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.mean(times) / total


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Norm stats ---
    data_min = float(np.load(os.path.join(NORM_DIR, "data_min.npy")))
    data_max = float(np.load(os.path.join(NORM_DIR, "data_max.npy")))
    print(f"Norm stats: [{data_min:.4f}, {data_max:.4f}]")

    # --- Real data ---
    with h5py.File(DATA_PATH, 'r') as f:
        outputs = np.array(f['tensor']).astype(np.float32)
    if outputs.ndim == 3:
        outputs = outputs[:, np.newaxis, :, :]
    real_test = outputs[9000:]
    # Shuffle so we get different samples each seed
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(real_test))
    real_test = real_test[idx]
    print(f"Real test data: {real_test.shape[0]} samples")

    # --- Shared noise ---
    C, H, W = DATA_SHAPE
    batch_size = 16
    rounds = (args.num_samples + batch_size - 1) // batch_size
    z_batches = []
    for r in range(rounds):
        n = min(batch_size, args.num_samples - r * batch_size)
        z_batches.append(th.randn(n, C, H, W))

    # --- Collect results: {name: {samples_denorm, time_per_sample, steps}} ---
    results = {}

    # Ground truth
    results['Ground Truth'] = {
        'samples': real_test[:args.num_samples],
        'steps': '-',
        'time': None,
    }

    # Teacher
    print("\nLoading teacher...")
    net = build_unet()
    teacher = VPDiffusionModel(network=net, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(TEACHER_CKPT, map_location='cpu', weights_only=True)
    teacher.network.load_state_dict(state['model_state_dict'])
    teacher.to(device).eval()
    print(f"Loaded teacher (raw weights) from {TEACHER_CKPT}")

    print(f"Generating {args.num_samples} teacher samples (DDIM, {DDIM_STEPS} steps)...")
    teacher_norm = sample_teacher(teacher, z_batches, device)
    teacher_denorm = denormalize(teacher_norm[:args.num_samples], data_min, data_max)

    print("Timing teacher...")
    teacher_time = time_model(
        lambda zb, d: sample_teacher(teacher, zb, d),
        z_batches, device, num_runs=args.timing_runs,
    )
    results['Teacher\n(50 DDIM steps)'] = {
        'samples': teacher_denorm,
        'steps': 50,
        'time': teacher_time,
    }
    del teacher
    th.cuda.empty_cache()

    # Students
    for cfg in STUDENT_CONFIGS:
        ckpt = find_latest_checkpoint(cfg['dir'])
        if ckpt is None:
            print(f"WARNING: No checkpoints in {cfg['dir']}, skipping")
            continue

        print(f"\nLoading {cfg['name']}...")
        net = build_unet()
        cm = MultistepConsistencyModel(
            network=net, student_steps=cfg['steps'], schedule_s=SCHEDULE_S,
        )
        state = th.load(ckpt, map_location='cpu', weights_only=True)
        cm.network.load_state_dict(state['model_state_dict'])
        if 'ema_state_dict' in state:
            cm.ema_network.load_state_dict(state['ema_state_dict'])
        epoch = state.get('epoch', '?')
        cm.to(device).eval()
        print(f"Loaded from {ckpt} (epoch {epoch})")

        print(f"Generating {args.num_samples} samples ({cfg['steps']} steps)...")
        student_norm = sample_student(cm, z_batches, device)
        student_denorm = denormalize(student_norm[:args.num_samples], data_min, data_max)

        print("Timing...")
        student_time = time_model(
            lambda zb, d, m=cm: sample_student(m, zb, d),
            z_batches, device, num_runs=args.timing_runs,
        )
        results[cfg['name']] = {
            'samples': student_denorm,
            'steps': cfg['steps'],
            'time': student_time,
        }
        del cm
        th.cuda.empty_cache()

    # ===========================================================
    # FIGURE 1: Sample grids + speedup table
    # ===========================================================
    print("\n--- Generating advisor_samples.png ---")
    model_names = list(results.keys())
    n_models = len(model_names)
    n_show = 8  # samples per row

    fig = plt.figure(figsize=(28, 4.2 * n_models + 2.5))
    gs = gridspec.GridSpec(n_models + 1, 1,
                           height_ratios=[1.0] * n_models + [0.6],
                           hspace=0.35)

    # Sample grid rows
    for row, name in enumerate(model_names):
        gs_row = gridspec.GridSpecFromSubplotSpec(1, n_show, subplot_spec=gs[row],
                                                  wspace=0.05)
        samples = results[name]['samples']
        for col in range(n_show):
            ax = fig.add_subplot(gs_row[0, col])
            if col < len(samples):
                ax.imshow(samples[col, 0], cmap='viridis')
            ax.axis('off')

        # Row label
        ax_label = fig.add_subplot(gs_row[0, 0])
        ax_label.set_title(name, fontsize=14, fontweight='bold',
                           loc='left', pad=12)

    # Speedup table at bottom
    ax_table = fig.add_subplot(gs[n_models])
    ax_table.axis('off')

    # Build table data
    col_labels = ['Model', 'Steps', 'Time/Sample (s)', 'Speedup']
    table_data = []
    for name in model_names:
        r = results[name]
        t = r['time']
        steps = str(r['steps'])
        if t is not None:
            time_str = f"{t:.4f}"
            speedup = f"{teacher_time / t:.1f}x" if teacher_time else "-"
        else:
            time_str = "-"
            speedup = "-"
        # Clean up name for table (remove newlines)
        clean_name = name.replace('\n', ' ')
        table_data.append([clean_name, steps, time_str, speedup])

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.8)

    # Style header row
    for col_idx in range(len(col_labels)):
        cell = table[0, col_idx]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for row_idx in range(1, len(table_data) + 1):
        color = '#ECF0F1' if row_idx % 2 == 0 else 'white'
        for col_idx in range(len(col_labels)):
            table[row_idx, col_idx].set_facecolor(color)

    fig.suptitle('Darcy Flow: Sample Quality & Inference Speed',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(os.path.join(args.output_dir, 'advisor_samples.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {args.output_dir}/advisor_samples.png")

    # ===========================================================
    # FIGURE 2: Histograms
    # ===========================================================
    print("\n--- Generating advisor_histograms.png ---")

    colors = {
        'Ground Truth': '#2C3E50',
        'Teacher\n(50 DDIM steps)': '#2980B9',
        'CD Student\n(4 steps)': '#E74C3C',
        'CD Student\n(8 steps)': '#E67E22',
        'CD Student\n(16 steps)': '#27AE60',
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: overlaid histograms
    ax = axes[0]
    for name in model_names:
        samples = results[name]['samples']
        clean_name = name.replace('\n', ' ')
        color = colors.get(name, 'gray')
        ax.hist(samples.flatten(), bins=150, alpha=0.5, density=True,
                label=clean_name, color=color, histtype='stepfilled')
    ax.set_xlabel('u(x, y)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Pixel Value Distribution (Overlaid)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: separate histograms stacked vertically
    ax2 = axes[1]
    for name in model_names:
        samples = results[name]['samples']
        clean_name = name.replace('\n', ' ')
        color = colors.get(name, 'gray')
        ax2.hist(samples.flatten(), bins=150, alpha=0.6, density=True,
                 label=clean_name, color=color, histtype='step', linewidth=2)
    ax2.set_xlabel('u(x, y)', fontsize=13)
    ax2.set_ylabel('Density', fontsize=13)
    ax2.set_title('Pixel Value Distribution (Line)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Darcy Flow: Distribution Comparison',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'advisor_histograms.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {args.output_dir}/advisor_histograms.png")

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
