"""
Measure the moment gap between teacher samples and student samples
(full inference chain, NOT training-time predictions).

This tells us:
  1. How large the moment loss would be if computed on actual samples
  2. What weight to use relative to CD loss (~0.03)

Usage:
    python scripts/measure_moment_gap.py --gpu 0 --n_samples 512
"""

import argparse
import glob
import os
import re
import numpy as np
import torch as th

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.consistency_models import MultistepConsistencyModel
from src.inference.samplers import MultistepCMSampler
from src.models.diffusion_utils import ddim_step


UNET_CFG = dict(
    dim=[1, 128, 128],
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

DATA_SHAPE = (1, 128, 128)
SCHEDULE_S = 0.008
STUDENT_STEPS = 16
DDIM_STEPS = 50
TEACHER_CKPT = "darcy_teacher/exp_1/saved_state/checkpoint_200.pt"


def find_latest_checkpoint(exp_dir):
    pattern = os.path.join(exp_dir, "saved_state", "checkpoint_*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    def epoch_num(path):
        m = re.search(r"checkpoint_(\d+)\.pt", path)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=epoch_num)


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


def sample_student(ckpt_path, initial_noise, device, batch_size=64):
    network = UNetModel(**UNET_CFG)
    model = MultistepConsistencyModel(
        network=network, student_steps=STUDENT_STEPS,
        schedule_s=SCHEDULE_S, infer=True,
    )
    state = th.load(ckpt_path, map_location="cpu", weights_only=True)
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
    return th.cat(all_samples, dim=0)


def compute_moments(samples):
    """Compute per-sample spatial mean and variance, return batch stats."""
    flat = samples.flatten(1)
    mu = flat.mean(dim=1)       # per-sample spatial mean
    var = flat.var(dim=1)        # per-sample spatial variance
    return mu, var


def moment_loss(mu_a, var_a, mu_b, var_b):
    """Same loss formulation as in objectives.py but on actual samples."""
    loss_mu = (mu_a.mean() - mu_b.mean()) ** 2 + (mu_a.var() - mu_b.var()) ** 2
    loss_var = (var_a.mean() - var_b.mean()) ** 2 + (var_a.var() - var_b.var()) ** 2
    return loss_mu.item(), loss_var.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--student_dir", type=str, default="darcy_student/exp_3",
                        help="Student experiment directory (baseline)")
    args = parser.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")

    # Fixed noise
    th.manual_seed(42)
    noise = th.randn(args.n_samples, *DATA_SHAPE)

    # Sample from teacher
    print(f"Sampling {args.n_samples} from teacher (50 DDIM steps)...")
    teacher_samples = sample_teacher(noise, device)
    mu_teacher, var_teacher = compute_moments(teacher_samples)

    # Sample from student
    ckpt = find_latest_checkpoint(args.student_dir)
    epoch = re.search(r"checkpoint_(\d+)", ckpt).group(1)
    print(f"Sampling {args.n_samples} from student {args.student_dir} (epoch {epoch}, 16 steps)...")
    student_samples = sample_student(ckpt, noise, device)
    mu_student, var_student = compute_moments(student_samples)

    # Compute moment losses
    loss_mu, loss_var = moment_loss(mu_student, var_student, mu_teacher, var_teacher)

    # Also compute vs real data moments for reference
    # (teacher moments are our actual target)

    print("\n" + "=" * 70)
    print("MOMENT STATISTICS (normalized [-1, 1] space)")
    print("=" * 70)
    print(f"{'':30s} {'Teacher':>12s} {'Student':>12s} {'Gap':>12s}")
    print("-" * 70)
    print(f"{'mean(spatial_mean)':30s} {mu_teacher.mean():12.6f} {mu_student.mean():12.6f} {abs(mu_teacher.mean() - mu_student.mean()):12.6f}")
    print(f"{'var(spatial_mean)':30s} {mu_teacher.var():12.6f} {mu_student.var():12.6f} {abs(mu_teacher.var() - mu_student.var()):12.6f}")
    print(f"{'mean(spatial_var)':30s} {var_teacher.mean():12.6f} {var_student.mean():12.6f} {abs(var_teacher.mean() - var_student.mean()):12.6f}")
    print(f"{'var(spatial_var)':30s} {var_teacher.var():12.6f} {var_student.var():12.6f} {abs(var_teacher.var() - var_student.var()):12.6f}")

    print("\n" + "=" * 70)
    print("MOMENT LOSS VALUES (same formula as training)")
    print("=" * 70)
    print(f"loss_mu (mean matching):     {loss_mu:.8f}")
    print(f"loss_var (variance matching): {loss_var:.8f}")
    print(f"loss_mu + loss_var:           {loss_mu + loss_var:.8f}")

    print("\n" + "=" * 70)
    print("WEIGHT RECOMMENDATIONS (CD loss ~ 0.03)")
    print("=" * 70)
    cd_loss = 0.03  # approximate
    for w in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        contrib_mu = w * loss_mu
        contrib_var = w * loss_var
        total = contrib_mu + contrib_var
        pct = total / cd_loss * 100
        print(f"  weight={w:5.1f}:  moment_contribution={total:.6f}  ({pct:.1f}% of CD loss)")

    print("\n" + "=" * 70)
    print("DISTRIBUTION SHAPE")
    print("=" * 70)
    print(f"Teacher spatial_mean:  min={mu_teacher.min():.4f}  max={mu_teacher.max():.4f}  std={mu_teacher.std():.4f}")
    print(f"Student spatial_mean:  min={mu_student.min():.4f}  max={mu_student.max():.4f}  std={mu_student.std():.4f}")
    print(f"Teacher spatial_var:   min={var_teacher.min():.4f}  max={var_teacher.max():.4f}  std={var_teacher.std():.4f}")
    print(f"Student spatial_var:   min={var_student.min():.4f}  max={var_student.max():.4f}  std={var_student.std():.4f}")


if __name__ == "__main__":
    main()
