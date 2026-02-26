"""
Comprehensive evaluation: Real vs Teacher vs Students (4/8/16 steps).

Produces 4 separate output files:
  1. sample_grids.png        — Visual sample comparison
  2. mean_std_fields.png     — Pixelwise mean & std fields
  3. power_spectral_density.png — Radially averaged 2D PSD
  4. sampling_speed.txt      — Timing comparison table

Usage:
    python scripts/evaluate_all.py
    python scripts/evaluate_all.py --num-samples 512 --output-dir eval_results
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

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.consistency_models import MultistepConsistencyModel
from src.models.diffusion_utils import ddim_step


# ============================================================
# Constants
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

# Default model configurations
TEACHER_CKPT = "darcy_teacher/exp_1/saved_state/checkpoint_200.pt"
TEACHER_NORM_DIR = "darcy_teacher/exp_1/saved_state"

STUDENT_CONFIGS = [
    {"name": "Student (4 steps)",  "dir": "darcy_student/exp_1/saved_state", "steps": 4},
    {"name": "Student (8 steps)",  "dir": "darcy_student/exp_2/saved_state", "steps": 8},
    {"name": "Student (16 steps)", "dir": "darcy_student/exp_3/saved_state", "steps": 16},
]


# ============================================================
# Helpers
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate all models")
    p.add_argument("--data-path", default="data/2D_DarcyFlow_beta1.0_Train.hdf5")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-dir", default="eval_results")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timing-runs", type=int, default=5)
    return p.parse_args()


def build_unet():
    return UNetModel(**UNET_CFG)


def find_latest_checkpoint(save_dir):
    """Find the highest-epoch checkpoint in a directory. Returns None if missing."""
    if not os.path.isdir(save_dir):
        return None
    ckpts = [f for f in os.listdir(save_dir)
             if f.startswith('checkpoint_') and f.endswith('.pt')]
    if not ckpts:
        return None
    ckpts.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
    return os.path.join(save_dir, ckpts[-1])


def load_norm_stats(search_dirs):
    """Load data_min/data_max from the first directory that has them."""
    for d in search_dirs:
        min_path = os.path.join(d, "data_min.npy")
        if os.path.exists(min_path):
            data_min = float(np.load(min_path))
            data_max = float(np.load(os.path.join(d, "data_max.npy")))
            print(f"Loaded norm stats from {d}")
            return data_min, data_max
    raise FileNotFoundError("Could not find data_min.npy in any search directory")


def denormalize(x_norm, data_min, data_max):
    """Convert from [-1,1] normalized space to physical units."""
    if isinstance(x_norm, th.Tensor):
        x_norm = x_norm.cpu().numpy()
    return (x_norm + 1.0) / 2.0 * (data_max - data_min) + data_min


def load_real_data(data_path, data_min, data_max, train_samples=9000):
    """Load HDF5 and return denormalized test data."""
    with h5py.File(data_path, 'r') as f:
        outputs = np.array(f['tensor']).astype(np.float32)
    if outputs.ndim == 3:
        outputs = outputs[:, np.newaxis, :, :]
    test_raw = outputs[train_samples:]
    return test_raw


# ============================================================
# Model loading & sampling
# ============================================================

def load_teacher(ckpt_path, device):
    """Load teacher VP diffusion model with raw weights."""
    net = build_unet()
    teacher = VPDiffusionModel(network=net, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(ckpt_path, map_location='cpu', weights_only=True)
    teacher.network.load_state_dict(state['model_state_dict'])
    print(f"Loaded teacher (raw weights) from {ckpt_path}")
    teacher.to(device)
    teacher.eval()
    return teacher


def sample_teacher(teacher, noise_batches, device):
    """Sample from teacher using 50 DDIM steps. Returns normalized tensor."""
    ts = th.linspace(1.0, 0.0, DDIM_STEPS + 1, device=device)
    all_samples = []
    with th.no_grad():
        for noise in noise_batches:
            z = noise.to(device)
            n = z.shape[0]
            for i in range(DDIM_STEPS):
                t_batch = th.full((n,), ts[i].item(), device=device)
                s_batch = th.full((n,), ts[i + 1].item(), device=device)
                x_hat = teacher.predict_x(z, t_batch)
                z = ddim_step(x_hat, z, t_batch, s_batch, SCHEDULE_S)
            all_samples.append(z.cpu())
    return th.cat(all_samples, dim=0)


def load_student(ckpt_path, student_steps, device):
    """Load student consistency model."""
    net = build_unet()
    cm = MultistepConsistencyModel(
        network=net, student_steps=student_steps, schedule_s=SCHEDULE_S,
    )
    state = th.load(ckpt_path, map_location='cpu', weights_only=True)
    cm.network.load_state_dict(state['model_state_dict'])
    if 'ema_state_dict' in state:
        cm.ema_network.load_state_dict(state['ema_state_dict'])
    epoch = state.get('epoch', '?')
    print(f"Loaded student ({student_steps} steps) from {ckpt_path} (epoch {epoch})")
    cm.to(device)
    cm.eval()
    return cm


def sample_student(model, noise_batches, device):
    """Sample from student consistency model. Returns normalized tensor."""
    all_samples = []
    with th.no_grad():
        for noise in noise_batches:
            z = noise.to(device)
            samples = model.sample(z)
            all_samples.append(samples.cpu())
    return th.cat(all_samples, dim=0)


# ============================================================
# Analysis
# ============================================================

def compute_radial_psd(samples):
    """
    Compute radially averaged 2D power spectral density.

    Args:
        samples: np.ndarray of shape (N, 1, H, W) in physical units.

    Returns:
        (wavenumbers, mean_psd): Arrays of shape (num_bins,).
    """
    N, _, H, W = samples.shape
    assert H == W, "Assumes square spatial domain"

    all_psd = []
    for i in range(N):
        field = samples[i, 0]  # (H, W)
        fft2 = np.fft.fft2(field)
        power = np.abs(np.fft.fftshift(fft2)) ** 2 / (H * W)

        # Radial distances from center
        cy, cx = H // 2, W // 2
        y_idx, x_idx = np.ogrid[:H, :W]
        r = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2).astype(int)

        max_r = min(cy, cx)
        psd = np.zeros(max_r)
        counts = np.zeros(max_r)
        for ri in range(max_r):
            mask = r == ri
            psd[ri] = power[mask].sum()
            counts[ri] = mask.sum()
        psd = np.where(counts > 0, psd / counts, 0)
        all_psd.append(psd)

    mean_psd = np.mean(all_psd, axis=0)
    wavenumbers = np.arange(len(mean_psd))
    return wavenumbers, mean_psd


def time_sampling(sample_fn, noise_batches, device, num_runs=5):
    """
    Time a sampling function. Returns average seconds per sample.

    Args:
        sample_fn: Callable that takes (noise_batches, device) and returns samples.
        noise_batches: List of noise tensors.
        device: Torch device.
        num_runs: Number of timed runs to average.
    """
    # Use a small subset for timing (2 batches max)
    timing_noise = noise_batches[:2]
    total_samples = sum(n.shape[0] for n in timing_noise)

    # Warmup
    sample_fn(timing_noise, device)
    if device != 'cpu':
        th.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        if device != 'cpu':
            th.cuda.synchronize()
        t0 = time.perf_counter()
        sample_fn(timing_noise, device)
        if device != 'cpu':
            th.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = np.mean(times)
    return avg_time / total_samples


# ============================================================
# Plotting
# ============================================================

def plot_sample_grids(results, output_path):
    """Output 1: Side-by-side sample grids (auto-scaled per image)."""
    n_cols = 8
    model_names = list(results.keys())
    n_rows = len(model_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(model_names):
        samples = results[name]['samples']
        for col in range(n_cols):
            if col < len(samples):
                axes[row, col].imshow(samples[col, 0], cmap='viridis')
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(name, fontsize=11, rotation=0,
                                          labelpad=100, va='center')

    plt.suptitle('Sample Comparison', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_mean_std_fields(results, output_path):
    """Output 2: Pixelwise mean and std fields."""
    model_names = list(results.keys())
    n_cols = len(model_names)

    # Compute mean and std fields
    mean_fields = {}
    std_fields = {}
    for name in model_names:
        s = results[name]['samples']
        mean_fields[name] = s.mean(axis=0)[0]  # (H, W)
        std_fields[name] = s.std(axis=0)[0]

    # Global ranges
    mean_vmin = min(m.min() for m in mean_fields.values())
    mean_vmax = max(m.max() for m in mean_fields.values())
    std_vmin = min(s.min() for s in std_fields.values())
    std_vmax = max(s.max() for s in std_fields.values())

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col, name in enumerate(model_names):
        im_mean = axes[0, col].imshow(
            mean_fields[name], cmap='viridis', vmin=mean_vmin, vmax=mean_vmax
        )
        axes[0, col].set_title(name, fontsize=10)
        axes[0, col].axis('off')

        im_std = axes[1, col].imshow(
            std_fields[name], cmap='magma', vmin=std_vmin, vmax=std_vmax
        )
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Mean Field', fontsize=12, rotation=0, labelpad=70, va='center')
    axes[1, 0].set_ylabel('Std Field', fontsize=12, rotation=0, labelpad=70, va='center')

    fig.colorbar(im_mean, ax=axes[0, :].tolist(), shrink=0.8, label='Mean u(x,y)')
    fig.colorbar(im_std, ax=axes[1, :].tolist(), shrink=0.8, label='Std u(x,y)')

    plt.suptitle('Pixelwise Mean & Standard Deviation', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_psd(results, output_path):
    """Output 3: Radially averaged power spectral density."""
    styles = {
        'Real Data':           {'color': 'black',  'ls': '-',  'lw': 2.0},
        'Teacher (50 steps)':  {'color': 'blue',   'ls': '-',  'lw': 1.5},
        'Student (4 steps)':   {'color': 'red',    'ls': '--', 'lw': 1.5},
        'Student (8 steps)':   {'color': 'orange',  'ls': '-.', 'lw': 1.5},
        'Student (16 steps)':  {'color': 'green',  'ls': ':',  'lw': 1.5},
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for name in results:
        k, psd = results[name]['psd']
        # Skip k=0 (DC component)
        mask = k > 0
        style = styles.get(name, {'color': 'gray', 'ls': '-', 'lw': 1.0})
        ax.loglog(k[mask], psd[mask], label=name, **style)

    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel('Power Spectral Density', fontsize=12)
    ax.set_title('Radially Averaged Power Spectral Density', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def write_timing_table(results, output_path):
    """Output 4: Sampling speed comparison."""
    lines = []
    header = f"{'Model':<25s} {'Steps':>6s} {'Time/sample (s)':>16s} {'Speedup':>10s}"
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    teacher_time = None
    for name in results:
        t = results[name].get('time_per_sample')
        if t is None:
            continue
        steps = results[name].get('steps', '-')
        if 'Teacher' in name:
            teacher_time = t

    for name in results:
        t = results[name].get('time_per_sample')
        if t is None:
            continue
        steps = results[name].get('steps', '-')
        if teacher_time and teacher_time > 0:
            speedup = f"{teacher_time / t:.1f}x"
        else:
            speedup = "1.0x" if 'Teacher' in name else "-"
        lines.append(f"{name:<25s} {str(steps):>6s} {t:>16.4f} {speedup:>10s}")

    table = "\n".join(lines)
    print("\n" + table)

    with open(output_path, 'w') as f:
        f.write(table + "\n")
    print(f"\nSaved {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # --- Norm stats ---
    search_dirs = [TEACHER_NORM_DIR] + [s['dir'] for s in STUDENT_CONFIGS]
    data_min, data_max = load_norm_stats(search_dirs)

    # --- Real data ---
    print("\n--- Loading real data ---")
    real_raw = load_real_data(args.data_path, data_min, data_max)
    print(f"Test data: {real_raw.shape[0]} samples, "
          f"range [{real_raw.min():.4f}, {real_raw.max():.4f}]")

    # --- Pre-generate shared noise ---
    C, H, W = DATA_SHAPE
    rounds = (args.num_samples + args.batch_size - 1) // args.batch_size
    noise_batches = []
    for r in range(rounds):
        n = min(args.batch_size, args.num_samples - r * args.batch_size)
        noise_batches.append(th.randn(n, C, H, W))

    # --- Results dict (ordered) ---
    results = {}

    # --- Real data entry ---
    # Subsample if we have more test data than num_samples
    n_real = min(len(real_raw), args.num_samples)
    real_subset = real_raw[:n_real]
    if real_subset.ndim == 3:
        real_subset = real_subset[:, np.newaxis, :, :]

    print(f"\nComputing real data PSD ({n_real} samples)...")
    k_real, psd_real = compute_radial_psd(real_subset)
    results['Real Data'] = {
        'samples': real_subset,
        'psd': (k_real, psd_real),
        'steps': '-',
    }

    # --- Teacher ---
    print("\n--- Teacher ---")
    if os.path.exists(TEACHER_CKPT):
        teacher = load_teacher(TEACHER_CKPT, device)

        print(f"Generating {args.num_samples} teacher samples ({DDIM_STEPS} aDDIM steps)...")
        teacher_samples_norm = sample_teacher(teacher, noise_batches, device)
        teacher_samples = denormalize(teacher_samples_norm[:args.num_samples], data_min, data_max)

        print("Computing teacher PSD...")
        k_t, psd_t = compute_radial_psd(teacher_samples)

        print("Timing teacher sampling...")
        t_per_sample = time_sampling(
            lambda nb, d: sample_teacher(teacher, nb, d),
            noise_batches, device, num_runs=args.timing_runs,
        )

        results['Teacher (50 steps)'] = {
            'samples': teacher_samples,
            'psd': (k_t, psd_t),
            'steps': DDIM_STEPS,
            'time_per_sample': t_per_sample,
        }

        del teacher
        th.cuda.empty_cache()
    else:
        print(f"WARNING: Teacher checkpoint not found at {TEACHER_CKPT}, skipping")

    # --- Students ---
    for cfg in STUDENT_CONFIGS:
        name = cfg['name']
        print(f"\n--- {name} ---")

        ckpt = find_latest_checkpoint(cfg['dir'])
        if ckpt is None:
            print(f"WARNING: No checkpoints found in {cfg['dir']}, skipping")
            continue

        model = load_student(ckpt, cfg['steps'], device)

        print(f"Generating {args.num_samples} samples ({cfg['steps']} steps)...")
        student_samples_norm = sample_student(model, noise_batches, device)
        student_samples = denormalize(student_samples_norm[:args.num_samples], data_min, data_max)

        print(f"Computing PSD...")
        k_s, psd_s = compute_radial_psd(student_samples)

        print(f"Timing sampling...")
        t_per_sample = time_sampling(
            lambda nb, d, m=model: sample_student(m, nb, d),
            noise_batches, device, num_runs=args.timing_runs,
        )

        results[name] = {
            'samples': student_samples,
            'psd': (k_s, psd_s),
            'steps': cfg['steps'],
            'time_per_sample': t_per_sample,
        }

        del model
        th.cuda.empty_cache()

    # --- Generate outputs ---
    print("\n" + "=" * 60)
    print("GENERATING OUTPUTS")
    print("=" * 60)

    plot_sample_grids(results, os.path.join(args.output_dir, "sample_grids.png"))
    plot_mean_std_fields(results, os.path.join(args.output_dir, "mean_std_fields.png"))
    plot_psd(results, os.path.join(args.output_dir, "power_spectral_density.png"))
    write_timing_table(results, os.path.join(args.output_dir, "sampling_speed.txt"))

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
