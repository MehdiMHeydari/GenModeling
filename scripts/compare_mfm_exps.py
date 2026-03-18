"""
Compare MFM experiments side-by-side: sample grids + histograms.

Experiments:
  - exp_5: gamma=0.5, max_norm=1.0 (baseline)
  - exp_7: gamma=0.5, grad_accum=4, max_norm=2000 (current)

Usage:
    python scripts/compare_mfm_exps.py --gpu 0
"""

import argparse
import glob
import os
import re
import numpy as np
import h5py
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.flow_models import MeanFlowMatching
from src.models.vp_diffusion import VPDiffusionModel
from src.inference.samplers import MeanSampler
from src.models.diffusion_utils import ddim_step


DATA_SHAPE = (1, 128, 128)
SCHEDULE_S = 0.008
DDIM_STEPS = 50
DATA_PATH = "data/2D_DarcyFlow_beta1.0_Train.hdf5"
STATS_DIR = "darcy_teacher/exp_1/saved_state"
TEACHER_CKPT = "darcy_teacher/exp_1/saved_state/checkpoint_200.pt"

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

MFM_EXPS = {
    "exp_5": {"dir": "darcy_mean_flow/exp_5/saved_state",
              "label": "MFM exp5\n(gamma=0.5, norm=1)",
              "steps": [1, 2, 4]},
    "exp_7": {"dir": "darcy_mean_flow/exp_7/saved_state",
              "label": "MFM exp7\n(gamma=0.5, accum=4, norm=2k)",
              "steps": [1, 2, 4]},
}


def find_latest_checkpoint(save_dir):
    ckpts = glob.glob(os.path.join(save_dir, "checkpoint_*.pt"))
    if not ckpts:
        return None, None
    def epoch_num(p):
        m = re.search(r"checkpoint_(\d+)\.pt", p)
        return int(m.group(1)) if m else -1
    best = max(ckpts, key=epoch_num)
    return best, epoch_num(best)


def denormalize(samples, data_min, data_max):
    if isinstance(samples, th.Tensor):
        samples = samples.numpy()
    return (samples + 1.0) / 2.0 * (data_max - data_min) + data_min


def sample_teacher(initial_noise, device):
    network = UNetModel(**UNET_CFG)
    teacher = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(TEACHER_CKPT, map_location="cpu", weights_only=True)
    teacher.network.load_state_dict(state["model_state_dict"])
    teacher.to(device).eval()

    ts = th.linspace(1.0, 0.0, DDIM_STEPS + 1, device=device)
    batch_size = 64
    all_samples = []
    with th.no_grad():
        for i in range(0, initial_noise.shape[0], batch_size):
            z = initial_noise[i:i+batch_size].to(device)
            n = z.shape[0]
            for step in range(DDIM_STEPS):
                t_batch = th.full((n,), ts[step].item(), device=device)
                s_batch = th.full((n,), ts[step + 1].item(), device=device)
                x_hat = teacher.predict_x(z, t_batch)
                z = ddim_step(x_hat, z, t_batch, s_batch, SCHEDULE_S)
            all_samples.append(z.cpu())

    del teacher, network
    th.cuda.empty_cache()
    return th.cat(all_samples, dim=0)


def sample_mfm(ckpt_path, initial_noise, device, n_steps=2):
    unet_cfg = dict(UNET_CFG)
    unet_cfg["use_future_time_emb"] = True
    network = UNetModel(**unet_cfg)
    model = MeanFlowMatching(network=network)
    state = th.load(ckpt_path, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.infer = True
    model.network.eval()

    sampler = MeanSampler(model)
    batch_size = 64
    all_samples = []
    t_span_kwargs = {"start": 0, "end": 1, "steps": n_steps + 1}
    with th.no_grad():
        for i in range(0, initial_noise.shape[0], batch_size):
            z = initial_noise[i:i+batch_size].to(device)
            samples = sampler.sample(z, t_span_kwargs=t_span_kwargs)
            all_samples.append(samples.cpu())

    del model, network
    th.cuda.empty_cache()
    return th.cat(all_samples, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_show", type=int, default=6)
    parser.add_argument("--mfm_steps", type=int, default=2,
                        help="Number of MFM sampling steps for grid/histogram")
    parser.add_argument("--output_dir", type=str, default="eval_mfm_compare")
    args = parser.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Norm stats
    data_min = float(np.load(os.path.join(STATS_DIR, "data_min.npy")))
    data_max = float(np.load(os.path.join(STATS_DIR, "data_max.npy")))

    # Ground truth
    with h5py.File(DATA_PATH, "r") as f:
        outputs = np.array(f["tensor"]).astype(np.float32)
    if outputs.ndim == 3:
        outputs = outputs[:, np.newaxis, :, :]
    real_denorm = outputs[9500:]

    # Fixed noise
    th.manual_seed(42)
    initial_noise = th.randn(args.n_samples, *DATA_SHAPE)

    # Collect results: {label: denorm_samples}
    results = {}
    results["Ground Truth"] = real_denorm[:args.n_samples]

    # Teacher
    print("Sampling teacher (50 DDIM)...")
    teacher_samples = sample_teacher(initial_noise, device)
    results["Teacher\n(50 DDIM)"] = denormalize(teacher_samples, data_min, data_max)

    # MFM experiments
    for name, info in MFM_EXPS.items():
        ckpt, epoch = find_latest_checkpoint(info["dir"])
        if ckpt is None:
            print(f"WARNING: No checkpoint for {name}, skipping")
            continue
        print(f"Sampling {name} (epoch {epoch}, {args.mfm_steps} steps)...")
        samples = sample_mfm(ckpt, initial_noise, device, n_steps=args.mfm_steps)
        label = f"{info['label']}\n(ep {epoch})"
        results[label] = denormalize(samples, data_min, data_max)

    # ============================================================
    # PLOT 1: Sample grid
    # ============================================================
    print("\nGenerating sample grid...")
    model_names = list(results.keys())
    n_rows = len(model_names)
    n_cols = args.n_show

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.5 * n_cols + 1.5, 2.5 * n_rows))
    plt.subplots_adjust(left=0.15, wspace=0.05, hspace=0.15)

    for row, name in enumerate(model_names):
        samples = results[name]
        for j in range(n_cols):
            axes[row, j].imshow(samples[j, 0], cmap="viridis")
            axes[row, j].axis("off")
        axes[row, 0].text(-0.15, 0.5, name,
                          transform=axes[row, 0].transAxes, fontsize=10,
                          fontweight="bold", va="center", ha="right")

    fig.suptitle(f"MFM Comparison ({args.mfm_steps}-step sampling)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    grid_path = os.path.join(args.output_dir, "mfm_samples.png")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {grid_path}")

    # ============================================================
    # PLOT 2: Histograms
    # ============================================================
    print("Generating histograms...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    color_list = ["gray", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    bins = np.linspace(0, real_denorm.max() * 1.05, 150)

    for i, (name, samples) in enumerate(results.items()):
        clean = name.replace('\n', ' ')
        c = color_list[i % len(color_list)]
        ax1.hist(samples.flatten(), bins=bins, alpha=0.3, density=True,
                 label=clean, color=c, histtype='stepfilled')
        counts, edges = np.histogram(samples.flatten(), bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax2.plot(centers, counts, label=clean, color=c, linewidth=1.5)

    for ax in [ax1, ax2]:
        ax.set_xlabel("u(x, y)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    ax1.set_title("Pixel Value Distribution (Overlaid)", fontsize=13)
    ax2.set_title("Pixel Value Distribution (Line)", fontsize=13)

    fig.suptitle("MFM: Distribution Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    hist_path = os.path.join(args.output_dir, "mfm_histogram.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {hist_path}")

    # Stats
    print("\n" + "=" * 75)
    print(f"{'':35s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print("-" * 75)
    for name, samples in results.items():
        clean = name.replace('\n', ' ')
        print(f"{clean:35s} {samples.mean():10.4f} {samples.std():10.4f} "
              f"{samples.min():10.4f} {samples.max():10.4f}")

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
