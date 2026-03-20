"""
Generate all presentation figures in one shot.
Each slide gets its own clearly-labeled PNG in the output folder.

Usage:
    python scripts/generate_presentation.py --gpu 7 --n_show 12 --n_hist 1000
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
from src.models.vp_diffusion import VPDiffusionModel
from src.models.consistency_models import MultistepConsistencyModel
from src.models.flow_models import MeanFlowMatching, RectifiedFlowMatching
from src.inference.samplers import MultistepCMSampler, MeanSampler, RectifiedFlowSampler
from src.models.diffusion_utils import ddim_step


# ============================================================
# Constants
# ============================================================
DATA_SHAPE = (1, 128, 128)
SCHEDULE_S = 0.008
DDIM_STEPS = 75
DATA_PATH = "data/2D_DarcyFlow_beta1.0_Train.hdf5"
STATS_DIR = "darcy_teacher/exp_1/saved_state"
TEACHER_CKPT = "darcy_teacher/exp_1/saved_state/checkpoint_200.pt"
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


# ============================================================
# Helpers
# ============================================================

def find_latest_checkpoint(save_dir):
    ckpts = glob.glob(os.path.join(save_dir, "checkpoint_*.pt"))
    if not ckpts:
        return None, None
    def epoch_num(p):
        m = re.search(r"checkpoint_(\d+)\.pt", p)
        return int(m.group(1)) if m else -1
    best = max(ckpts, key=epoch_num)
    return best, epoch_num(best)


def find_latest_pd_checkpoint(save_dir):
    ckpts = glob.glob(os.path.join(save_dir, "pd_round*_steps*.pt"))
    if not ckpts:
        return None, None, None
    def round_num(p):
        m = re.search(r"pd_round(\d+)_steps(\d+)\.pt", p)
        return int(m.group(1)) if m else -1
    best = max(ckpts, key=round_num)
    m = re.search(r"pd_round(\d+)_steps(\d+)\.pt", best)
    return best, int(m.group(1)), int(m.group(2))


def find_all_pd_checkpoints(save_dir):
    """Return list of (ckpt_path, round_num, student_steps) sorted by round."""
    ckpts = glob.glob(os.path.join(save_dir, "pd_round*_steps*.pt"))
    if not ckpts:
        return []
    results = []
    for c in ckpts:
        m = re.search(r"pd_round(\d+)_steps(\d+)\.pt", c)
        if m:
            results.append((c, int(m.group(1)), int(m.group(2))))
    return sorted(results, key=lambda x: x[1])


def denormalize(samples, data_min, data_max):
    if isinstance(samples, th.Tensor):
        samples = samples.cpu().numpy()
    return (samples + 1.0) / 2.0 * (data_max - data_min) + data_min


def sample_teacher(initial_noise, device, batch_size=64):
    network = UNetModel(**UNET_CFG)
    teacher = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(TEACHER_CKPT, map_location="cpu", weights_only=True)
    teacher.network.load_state_dict(state["model_state_dict"])
    teacher.to(device).eval()

    ts = th.linspace(1.0, 0.0, DDIM_STEPS + 1, device=device)
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


def sample_cd(exp_dir, initial_noise, device, batch_size=64):
    ckpt, epoch = find_latest_checkpoint(os.path.join(exp_dir, "saved_state"))
    if ckpt is None:
        return None, None
    network = UNetModel(**UNET_CFG)
    model = MultistepConsistencyModel(
        network=network, student_steps=16, schedule_s=SCHEDULE_S, infer=True,
    )
    state = th.load(ckpt, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    if "ema_state_dict" in state:
        model.ema_network.load_state_dict(state["ema_state_dict"])
    model.to(device).eval()

    sampler = MultistepCMSampler(model)
    all_samples = []
    with th.no_grad():
        for i in range(0, initial_noise.shape[0], batch_size):
            z = initial_noise[i:i+batch_size].to(device)
            samples = sampler.sample(z)
            all_samples.append(samples.cpu())

    del model, network
    th.cuda.empty_cache()
    return th.cat(all_samples, dim=0), epoch


def sample_pd_checkpoint(ckpt_path, student_steps, initial_noise, device):
    network = UNetModel(**UNET_CFG)
    model = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(ckpt_path, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    ts = th.linspace(1.0, 0.0, student_steps + 1, device=device)
    all_samples = []
    with th.no_grad():
        for i in range(0, initial_noise.shape[0], 64):
            z = initial_noise[i:i+64].to(device)
            n = z.shape[0]
            for step in range(student_steps):
                t_batch = th.full((n,), ts[step].item(), device=device).clamp(1e-4, 1 - 1e-4)
                s_batch = th.full((n,), ts[step + 1].item(), device=device).clamp(0, 1 - 1e-4)
                x_hat = model.predict_x(z, t_batch, use_ema=True)
                z = ddim_step(x_hat, z, t_batch, s_batch, SCHEDULE_S)
            all_samples.append(z.cpu())

    del model, network
    th.cuda.empty_cache()
    return th.cat(all_samples, dim=0)


def sample_mfm(exp_dir, initial_noise, device, n_steps=2, batch_size=64):
    ckpt, epoch = find_latest_checkpoint(os.path.join(exp_dir, "saved_state"))
    if ckpt is None:
        return None, None
    unet_cfg = dict(UNET_CFG)
    unet_cfg["use_future_time_emb"] = True
    network = UNetModel(**unet_cfg)
    model = MeanFlowMatching(network=network)
    state = th.load(ckpt, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.infer = True
    model.network.eval()

    sampler = MeanSampler(model)
    t_span_kwargs = {"start": 0, "end": 1, "steps": n_steps + 1}
    all_samples = []
    with th.no_grad():
        for i in range(0, initial_noise.shape[0], batch_size):
            z = initial_noise[i:i+batch_size].to(device)
            samples = sampler.sample(z, t_span_kwargs=t_span_kwargs)
            all_samples.append(samples.cpu())

    del model, network
    th.cuda.empty_cache()
    return th.cat(all_samples, dim=0), epoch


def sample_rf(ckpt_path, initial_noise, device, n_steps=5, batch_size=64):
    network = UNetModel(**UNET_CFG)
    model = RectifiedFlowMatching(network=network, add_heavy_noise=False)
    state = th.load(ckpt_path, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.infer = True
    model.network.eval()

    sampler = RectifiedFlowSampler(model)
    t_span_kwargs = {"start": 0, "end": 1, "steps": n_steps + 1}
    all_samples = []
    with th.no_grad():
        for i in range(0, initial_noise.shape[0], batch_size):
            z = initial_noise[i:i+batch_size].to(device)
            samples = sampler.sample(z, t_span_kwargs=t_span_kwargs)
            all_samples.append(samples.cpu())

    del model, network
    th.cuda.empty_cache()
    return th.cat(all_samples, dim=0)


# ============================================================
# Plotting helpers
# ============================================================

def plot_sample_grid(rows, n_show, title, save_path):
    """
    rows: list of (label, samples_np) where samples_np is [N, 1, H, W]
    """
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_show,
                             figsize=(1.8 * n_show + 2, 2.5 * n_rows))
    plt.subplots_adjust(left=0.12, wspace=0.03, hspace=0.15)

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, (label, samples) in enumerate(rows):
        for j in range(n_show):
            axes[i, j].imshow(samples[j, 0], cmap=CMAP)
            axes[i, j].axis("off")
        axes[i, 0].text(-0.15, 0.5, label,
                        transform=axes[i, 0].transAxes, fontsize=10,
                        fontweight="bold", va="center", ha="right")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_histogram(data_dict, title, save_path, bins=None):
    """
    data_dict: dict of {label: flat_array}.
    First entry is rendered as a filled shaded area (ground truth),
    remaining entries as line plots.
    """
    color_list = ["tab:blue", "tab:red", "tab:orange",
                  "tab:green", "tab:purple", "tab:cyan", "tab:pink"]

    if bins is None:
        all_vals = np.concatenate(list(data_dict.values()))
        bins = np.linspace(0, np.percentile(all_vals, 99.5) * 1.05, 150)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, data) in enumerate(data_dict.items()):
        counts, edges = np.histogram(data, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        if i == 0:
            # Ground truth as shaded area
            ax.fill_between(centers, counts, alpha=0.3, color="gray", label=label)
            ax.plot(centers, counts, color="gray", linewidth=1.0, alpha=0.5)
        else:
            c = color_list[(i - 1) % len(color_list)]
            ax.plot(centers, counts, label=label, color=c, linewidth=1.5)

    ax.set_xlabel("u(x, y)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_rf_sidebyside(rf_samples_dict, reflow_samples_dict, n_show, save_path):
    """
    Side-by-side grid: RF (Round 1) on left, Reflow (Round 2) on right.
    Each dict maps step_count -> samples_np [N, 1, H, W].
    """
    step_counts = sorted(rf_samples_dict.keys())
    n_rows = len(step_counts)

    fig, axes = plt.subplots(n_rows, n_show * 2 + 1,
                             figsize=(1.4 * (n_show * 2 + 1), 1.6 * n_rows))
    plt.subplots_adjust(wspace=0.03, hspace=0.08)

    for r, steps in enumerate(step_counts):
        # Left panel: RF
        rf_data = rf_samples_dict.get(steps)
        for j in range(n_show):
            ax = axes[r, j]
            if rf_data is not None:
                ax.imshow(rf_data[j, 0], cmap=CMAP)
            ax.axis("off")

        # Separator column
        axes[r, n_show].axis("off")

        # Right panel: Reflow
        reflow_data = reflow_samples_dict.get(steps)
        for j in range(n_show):
            ax = axes[r, n_show + 1 + j]
            if reflow_data is not None:
                ax.imshow(reflow_data[j, 0], cmap=CMAP)
            ax.axis("off")

        # Row label
        label = f"{steps} step{'s' if steps > 1 else ''}"
        axes[r, 0].text(-0.15, 0.5, label,
                        transform=axes[r, 0].transAxes, fontsize=9,
                        fontweight="bold", va="center", ha="right")

    # Column titles
    fig.text(0.25, 1.01, "Rectified Flow (Round 1)",
             ha="center", fontsize=13, fontweight="bold")
    fig.text(0.75, 1.01, "Reflow (Round 2)",
             ha="center", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--n_show", type=int, default=12,
                        help="Number of sample images per row in grids")
    parser.add_argument("--n_hist", type=int, default=500,
                        help="Number of samples for histograms")
    parser.add_argument("--output_dir", type=str, default="presentation")
    parser.add_argument("--skip_to", type=int, default=2,
                        help="Start from this slide number (2-6)")
    parser.add_argument("--slides", type=str, default=None,
                        help="Comma-separated slide numbers to run, e.g. '4,5'")
    args = parser.parse_args()

    # Parse --slides into a set; if not given, fall back to skip_to logic
    if args.slides:
        args.active_slides = set(int(s) for s in args.slides.split(","))
    else:
        args.active_slides = None

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load norm stats and ground truth
    data_min = float(np.load(os.path.join(STATS_DIR, "data_min.npy")))
    data_max = float(np.load(os.path.join(STATS_DIR, "data_max.npy")))

    with h5py.File(DATA_PATH, "r") as f:
        outputs = np.array(f["tensor"]).astype(np.float32)
    if outputs.ndim == 3:
        outputs = outputs[:, np.newaxis, :, :]
    real_denorm = outputs[9500:]  # test set

    # Fixed noise
    th.manual_seed(42)
    n_total = max(args.n_show, args.n_hist)
    initial_noise = th.randn(n_total, *DATA_SHAPE)

    gt_row = ("Ground Truth", real_denorm[:args.n_show])

    def should_run(slide_num):
        if args.active_slides:
            return slide_num in args.active_slides
        return args.skip_to <= slide_num

    # ===========================================================
    # TEACHER (only needed for slides 2, 3, 6)
    # ===========================================================
    teacher_slides = {2, 3, 6}
    needs_teacher = any(should_run(s) for s in teacher_slides)
    if needs_teacher:
        print("\n[Teacher] Sampling 75 DDIM steps...")
        teacher_samples = sample_teacher(initial_noise, device)
        teacher_denorm = denormalize(teacher_samples, data_min, data_max)
        teacher_row = ("Teacher\n(75 DDIM)", teacher_denorm[:args.n_show])
    else:
        teacher_denorm = None
        teacher_row = None

    # ===========================================================
    # SLIDE 2: Progressive Distillation
    # ===========================================================
    if should_run(2):
        print("\n[Slide 2] Progressive Distillation...")
        pd_ckpts = find_all_pd_checkpoints("darcy_pd/exp_1/saved_state")
        if pd_ckpts:
            pd_rows = [gt_row, teacher_row]
            pd_hist = {"Ground Truth": real_denorm.flatten(),
                       "Teacher (75 DDIM)": teacher_denorm[:args.n_hist].flatten()}
            for ckpt_path, round_num, steps in pd_ckpts:
                print(f"  PD round {round_num} ({steps} steps)...")
                samples = sample_pd_checkpoint(ckpt_path, steps, initial_noise, device)
                pd_denorm = denormalize(samples, data_min, data_max)
                pd_rows.append((f"PD\n({steps} steps)", pd_denorm[:args.n_show]))
                pd_hist[f"PD ({steps} steps)"] = pd_denorm[:args.n_hist].flatten()
            plot_sample_grid(
                pd_rows, args.n_show,
                "Progressive Distillation",
                os.path.join(args.output_dir, "slide2_progressive_distillation.png"),
            )
            plot_histogram(
                pd_hist,
                "Progressive Distillation: Distribution",
                os.path.join(args.output_dir, "slide2_pd_histogram.png"),
            )
        else:
            print("  WARNING: No PD checkpoints found, skipping")

    # ===========================================================
    # SLIDE 3: Consistency Distillation (baseline, no moment)
    # ===========================================================
    if should_run(3):
        print("\n[Slide 3] Consistency Distillation (exp 3 baseline)...")
        cd_result = sample_cd("darcy_student/exp_3", initial_noise, device)
        if cd_result[0] is not None:
            cd_samples, cd_epoch = cd_result
            cd_denorm = denormalize(cd_samples, data_min, data_max)
            plot_sample_grid(
                [gt_row, teacher_row,
                 ("Consistency Distillation\n(16 steps)", cd_denorm[:args.n_show])],
                args.n_show,
                "Consistency Distillation",
                os.path.join(args.output_dir, "slide3_consistency_distillation.png"),
            )
            plot_histogram(
                {"Ground Truth": real_denorm.flatten(),
                 "Teacher (75 DDIM)": teacher_denorm[:args.n_hist].flatten(),
                 "CD baseline (16 steps)": cd_denorm[:args.n_hist].flatten()},
                "Consistency Distillation: Distribution",
                os.path.join(args.output_dir, "slide3_cd_histogram.png"),
            )
        else:
            print("  WARNING: No CD exp_3 checkpoints found, skipping")

    # ===========================================================
    # SLIDE 4: Rectified Flow (side-by-side, steps 1-10)
    # ===========================================================
    if should_run(4):
        print("\n[Slide 4] Rectified Flow...")
        rf_ckpt = "darcy_rectified_flow/exp_1/saved_state/checkpoint_799.pt"
        reflow_ckpt = "darcy_rectified_flow_reflow/exp_1/saved_state/checkpoint_399.pt"

        rf_dict = {}   # step_count -> denorm samples
        reflow_dict = {}
        rf_hist = {"Ground Truth": real_denorm.flatten()}
        has_rf = False

        if os.path.exists(rf_ckpt):
            for n_steps in range(1, 11):
                print(f"  RF {n_steps}-step...")
                rf_samples = sample_rf(rf_ckpt, initial_noise, device, n_steps=n_steps)
                rf_denorm = denormalize(rf_samples, data_min, data_max)
                rf_dict[n_steps] = rf_denorm[:args.n_show]
                if n_steps in [1, 5, 10]:
                    rf_hist[f"RF ({n_steps} step{'s' if n_steps > 1 else ''})"] = rf_denorm[:args.n_hist].flatten()
            has_rf = True

        if os.path.exists(reflow_ckpt):
            for n_steps in range(1, 11):
                print(f"  Reflow {n_steps}-step...")
                reflow_samples = sample_rf(reflow_ckpt, initial_noise, device, n_steps=n_steps)
                reflow_denorm = denormalize(reflow_samples, data_min, data_max)
                reflow_dict[n_steps] = reflow_denorm[:args.n_show]
                if n_steps in [1, 5, 10]:
                    rf_hist[f"Reflow ({n_steps} step{'s' if n_steps > 1 else ''})"] = reflow_denorm[:args.n_hist].flatten()
            has_rf = True

        if has_rf:
            plot_rf_sidebyside(
                rf_dict, reflow_dict, args.n_show,
                os.path.join(args.output_dir, "slide4_rectified_flow.png"),
            )
            plot_histogram(
                rf_hist,
                "Rectified Flow: Distribution",
                os.path.join(args.output_dir, "slide4_rf_histogram.png"),
            )
        else:
            print("  WARNING: No RF checkpoints found, skipping")

    # ===========================================================
    # SLIDE 5: Mean Flow Matching
    # ===========================================================
    if should_run(5):
        print("\n[Slide 5] Mean Flow Matching...")
        mfm_rows = [gt_row]
        mfm_hist = {"Ground Truth": real_denorm.flatten()}
        has_mfm = False

        for exp_name, exp_dir, label in [
            ("exp_5", "darcy_mean_flow/exp_5", "MFM exp5\n(gamma=0.5, norm=1)"),
            ("exp_7", "darcy_mean_flow/exp_7", "MFM exp7\n(gamma=0.5, accum=4)"),
        ]:
            for n_steps in [2, 4, 16]:
                result = sample_mfm(exp_dir, initial_noise, device, n_steps=n_steps)
                if result[0] is not None:
                    samples, epoch = result
                    mfm_denorm = denormalize(samples, data_min, data_max)
                    row_label = f"{label}\n({n_steps}-step)"
                    mfm_rows.append((row_label, mfm_denorm[:args.n_show]))
                    if n_steps in [2, 16]:
                        clean = label.replace("\n", " ")
                        mfm_hist[f"{clean} ({n_steps}-step)"] = mfm_denorm[:args.n_hist].flatten()
                    has_mfm = True

        if has_mfm:
            plot_sample_grid(
                mfm_rows, args.n_show,
                "Mean Flow Matching",
                os.path.join(args.output_dir, "slide5_mean_flow_matching.png"),
            )
            plot_histogram(
                mfm_hist,
                "Mean Flow Matching: Distribution",
                os.path.join(args.output_dir, "slide5_mfm_histogram.png"),
            )
        else:
            print("  WARNING: No MFM checkpoints found, skipping")

    # ===========================================================
    # SLIDE 6: Moment Matching (our contribution)
    # ===========================================================
    if should_run(6):
        print("\n[Slide 6] Moment Matching...")
        moment_exps = {
            "exp_3":  ("darcy_student/exp_3",  "CD baseline\n(no moment)"),
            "exp_18": ("darcy_student/exp_18", "mu=8, var=150"),
            "exp_19": ("darcy_student/exp_19", "mu=2, var=150"),
            "exp_20": ("darcy_student/exp_20", "mu=4, var=150"),
            "exp_21": ("darcy_student/exp_21", "mu=4, var=200"),
            "exp_22": ("darcy_student/exp_22", "mu=16, var=150"),
        }

        moment_rows = [gt_row, teacher_row]
        moment_hist = {"Ground Truth": real_denorm.flatten(),
                       "Teacher (75 DDIM)": teacher_denorm[:args.n_hist].flatten()}
        has_moment = False

        for name, (exp_dir, label) in moment_exps.items():
            result = sample_cd(exp_dir, initial_noise, device)
            if result[0] is not None:
                samples, epoch = result
                exp_denorm = denormalize(samples, data_min, data_max)
                moment_rows.append((label, exp_denorm[:args.n_show]))
                clean = label.replace("\n", " ")
                moment_hist[clean] = exp_denorm[:args.n_hist].flatten()
                has_moment = True
            else:
                print(f"  WARNING: No checkpoint for {name}, skipping")

        if has_moment:
            plot_sample_grid(
                moment_rows, args.n_show,
                "Moment Matching Regularization",
                os.path.join(args.output_dir, "slide6_moment_matching.png"),
            )
            plot_histogram(
                moment_hist,
                "Moment Matching: Distribution Comparison",
                os.path.join(args.output_dir, "slide6_moment_histogram.png"),
            )

    # ===========================================================
    # Done
    # ===========================================================
    print(f"\nAll figures saved to {args.output_dir}/")
    print("Files:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
