"""
Diagnose whether the moment loss has any meaningful gradient signal
compared to the CD loss. Loads a checkpoint, runs one batch, and reports:
  - CD loss magnitude
  - Moment loss (mu and var) magnitude
  - Gradient norms from each loss term separately
  - Whether x_hat_online batch statistics already match x

Usage:
    python scripts/diagnose_moment_loss.py --exp_dir darcy_student/exp_6 --gpu 0
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
from src.utils.dataloader import get_darcy_loader
from src.utils.dataset import DATASETS
from src.models.diffusion_utils import (
    alpha_t as _alpha_t, sigma_t as _sigma_t, snr as _snr,
    q_sample, ddim_step, inv_ddim, addim_step, _broadcast_to_spatial,
)

import math

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

SCHEDULE_S = 0.008
STUDENT_STEPS = 16
TEACHER_CKPT = "darcy_teacher/exp_1/saved_state/checkpoint_200.pt"
DATA_PATH = "data/2D_DarcyFlow_beta1.0_Train.hdf5"


def find_latest_checkpoint(exp_dir):
    pattern = os.path.join(exp_dir, "saved_state", "checkpoint_*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    def epoch_num(path):
        m = re.search(r"checkpoint_(\d+)\.pt", path)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=epoch_num)


def get_grad_norm(model):
    total = 0.0
    for p in model.network.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="darcy_student/exp_6")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")

    # Load data
    train_loader, _, _ = get_darcy_loader(
        data_path=DATA_PATH, batch_size=64,
        dataset_cls=DATASETS["VF_FM"], train_samples=9000,
        save_dir="/tmp/diag",
    )

    # Load teacher
    teacher_net = UNetModel(**UNET_CFG)
    teacher = VPDiffusionModel(network=teacher_net, schedule_s=SCHEDULE_S, infer=True)
    t_state = th.load(TEACHER_CKPT, map_location="cpu", weights_only=True)
    teacher.network.load_state_dict(t_state["model_state_dict"])
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Load student
    ckpt_path = find_latest_checkpoint(args.exp_dir)
    print(f"Loading student from {ckpt_path}")
    student_net = UNetModel(**UNET_CFG)
    model = MultistepConsistencyModel(
        network=student_net, student_steps=STUDENT_STEPS,
        schedule_s=SCHEDULE_S, ema_rate=0.9999,
    )
    state = th.load(ckpt_path, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    if "ema_state_dict" in state:
        model.ema_network.load_state_dict(state["ema_state_dict"])
    model.to(device)
    model.network.train()

    # Get one batch
    batch = next(iter(train_loader))
    _, x = batch
    x = x.to(device)
    B = x.shape[0]
    d = float(x[0].numel())

    # Simulate CD forward pass (same as MultistepCDLoss.__call__)
    # Use a fixed high iteration count so warmup is past
    N_teacher = 1280
    N_per_segment = max(1, round(N_teacher / STUDENT_STEPS))
    T_total = N_per_segment * STUDENT_STEPS

    eps = th.randn_like(x)
    step = th.randint(0, STUDENT_STEPS, (B,), device=device)
    n_rel = th.randint(1, N_per_segment + 1, (B,), device=device)

    t_step = step.float() / STUDENT_STEPS
    t = t_step + n_rel.float() / T_total
    s = t - 1.0 / T_total
    t = t.clamp(1e-5, 1.0 - 1e-5)
    s = s.clamp(1e-5, 1.0 - 1e-5)
    t_step = t_step.clamp(0.0, 1.0 - 1e-5)

    z_t, _ = q_sample(x, t, noise=eps, s=SCHEDULE_S)

    with th.no_grad():
        x_teacher = teacher.predict_x(z_t, t)
    x_var = 0.75 * (x_teacher - x).flatten(1).pow(2).sum(1) / d
    with th.no_grad():
        z_s = addim_step(x_teacher, z_t, x_var, t, s, SCHEDULE_S)

    x_hat_online = model.predict_x(z_t, t, use_ema=False)

    with th.no_grad():
        x_hat_target = model.predict_x(z_s, s, use_ema=True)
        z_ref_t_step = ddim_step(x_hat_target, z_s, s, t_step, SCHEDULE_S)
        x_ref = inv_ddim(z_ref_t_step, z_t, t, t_step, SCHEDULE_S)

    # === CD Loss ===
    x_diff = x_ref.detach() - x_hat_online
    w_t = _snr(t, SCHEDULE_S) + 1.0
    w_t = _broadcast_to_spatial(w_t, x)
    huber_eps = 1e-4
    cd_loss = th.mean(w_t * (th.sqrt(x_diff ** 2 + huber_eps ** 2) - huber_eps))

    # === Moment Loss ===
    pred_flat = x_hat_online.flatten(1)
    real_flat = x.detach().flatten(1)
    mu_pred = pred_flat.mean(dim=1)
    mu_real = real_flat.mean(dim=1)
    var_pred = pred_flat.var(dim=1)
    var_real = real_flat.var(dim=1)

    loss_mu = (mu_pred.mean() - mu_real.mean()) ** 2 + (mu_pred.var() - mu_real.var()) ** 2
    loss_var = (var_pred.mean() - var_real.mean()) ** 2 + (var_pred.var() - var_real.var()) ** 2

    print("\n" + "=" * 70)
    print("LOSS MAGNITUDES")
    print("=" * 70)
    print(f"CD loss:          {cd_loss.item():.6f}")
    print(f"Moment loss (mu): {loss_mu.item():.10f}")
    print(f"Moment loss (var):{loss_var.item():.10f}")
    print(f"")
    print(f"Ratio CD / moment_mu:  {cd_loss.item() / (loss_mu.item() + 1e-15):.1f}x")
    print(f"Ratio CD / moment_var: {cd_loss.item() / (loss_var.item() + 1e-15):.1f}x")
    print(f"")
    print(f"With weight=0.1:  moment_var contribution = {0.1 * loss_var.item():.10f}")
    print(f"With weight=1.0:  moment_var contribution = {1.0 * loss_var.item():.10f}")
    print(f"CD loss is {cd_loss.item() / (1.0 * loss_var.item() + 1e-15):.0f}x larger even with weight=1.0")

    # === Batch Statistics Comparison ===
    print("\n" + "=" * 70)
    print("BATCH STATISTICS (are predictions already matching real data?)")
    print("=" * 70)
    print(f"{'':25s} {'Predictions':>15s} {'Real Data':>15s} {'Gap':>15s}")
    print("-" * 70)
    print(f"{'mean(spatial_mean)':25s} {mu_pred.mean().item():15.6f} {mu_real.mean().item():15.6f} {abs(mu_pred.mean().item() - mu_real.mean().item()):15.6f}")
    print(f"{'var(spatial_mean)':25s}  {mu_pred.var().item():15.6f} {mu_real.var().item():15.6f} {abs(mu_pred.var().item() - mu_real.var().item()):15.6f}")
    print(f"{'mean(spatial_var)':25s}  {var_pred.mean().item():15.6f} {var_real.mean().item():15.6f} {abs(var_pred.mean().item() - var_real.mean().item()):15.6f}")
    print(f"{'var(spatial_var)':25s}   {var_pred.var().item():15.6f} {var_real.var().item():15.6f} {abs(var_pred.var().item() - var_real.var().item()):15.6f}")

    # === Gradient Norms ===
    print("\n" + "=" * 70)
    print("GRADIENT NORMS (how much each loss moves the weights)")
    print("=" * 70)

    # CD gradients
    model.network.zero_grad()
    cd_loss.backward(retain_graph=True)
    cd_grad_norm = get_grad_norm(model)

    # Moment var gradients
    model.network.zero_grad()
    loss_var.backward(retain_graph=True)
    var_grad_norm = get_grad_norm(model)

    # Moment mu gradients
    model.network.zero_grad()
    loss_mu.backward()
    mu_grad_norm = get_grad_norm(model)

    print(f"CD loss grad norm:          {cd_grad_norm:.6f}")
    print(f"Moment var grad norm:       {var_grad_norm:.6f}")
    print(f"Moment mu grad norm:        {mu_grad_norm:.6f}")
    print(f"")
    print(f"Weighted var (w=0.1):       {0.1 * var_grad_norm:.6f}  ({0.1 * var_grad_norm / (cd_grad_norm + 1e-15) * 100:.2f}% of CD)")
    print(f"Weighted var (w=1.0):       {1.0 * var_grad_norm:.6f}  ({1.0 * var_grad_norm / (cd_grad_norm + 1e-15) * 100:.2f}% of CD)")
    print(f"Weighted var (w=10.0):      {10.0 * var_grad_norm:.6f}  ({10.0 * var_grad_norm / (cd_grad_norm + 1e-15) * 100:.2f}% of CD)")
    print(f"Weighted var (w=100.0):     {100.0 * var_grad_norm:.6f}  ({100.0 * var_grad_norm / (cd_grad_norm + 1e-15) * 100:.2f}% of CD)")

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    ratio = cd_grad_norm / (var_grad_norm + 1e-15)
    if ratio > 100:
        print(f"The CD gradient is {ratio:.0f}x larger than the moment gradient.")
        print(f"The moment loss is effectively invisible to the optimizer.")
        print(f"You would need weight >= {ratio:.0f} for the moment loss to matter.")
    elif ratio > 10:
        print(f"The CD gradient is {ratio:.0f}x larger. Moment loss is weak but present.")
    else:
        print(f"Gradients are comparable (ratio {ratio:.1f}x). Moment loss should have effect.")


if __name__ == "__main__":
    main()
