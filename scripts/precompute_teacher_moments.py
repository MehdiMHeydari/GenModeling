"""
Pre-compute teacher distribution moments from 1000 DDIM samples and save to disk.
These are used as targets for the sampling-based moment loss during CD training.

Usage:
    python scripts/precompute_teacher_moments.py --gpu 0 --n_samples 1000
"""

import argparse
import os
import torch as th
import numpy as np

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
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

SCHEDULE_S = 0.008
DDIM_STEPS = 50
TEACHER_CKPT = "darcy_teacher/exp_1/saved_state/checkpoint_200.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str,
                        default="darcy_teacher/exp_1/saved_state")
    args = parser.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")

    # Load teacher
    network = UNetModel(**UNET_CFG)
    teacher = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(TEACHER_CKPT, map_location="cpu", weights_only=True)
    teacher.network.load_state_dict(state["model_state_dict"])
    teacher.to(device).eval()

    # Sample
    th.manual_seed(123)
    ts = th.linspace(1.0, 0.0, DDIM_STEPS + 1, device=device)
    batch_size = 64
    all_samples = []

    print(f"Sampling {args.n_samples} from teacher (50 DDIM steps)...")
    with th.no_grad():
        for i in range(0, args.n_samples, batch_size):
            n = min(batch_size, args.n_samples - i)
            z = th.randn(n, 1, 128, 128, device=device)
            for step in range(DDIM_STEPS):
                t_batch = th.full((n,), ts[step].item(), device=device)
                s_batch = th.full((n,), ts[step + 1].item(), device=device)
                x_hat = teacher.predict_x(z, t_batch)
                z = ddim_step(x_hat, z, t_batch, s_batch, SCHEDULE_S)
            all_samples.append(z.cpu())
            print(f"  {i + n}/{args.n_samples}")

    samples = th.cat(all_samples, dim=0)

    # Compute moments
    flat = samples.flatten(1)
    mu = flat.mean(dim=1)    # per-sample spatial mean
    var = flat.var(dim=1)     # per-sample spatial variance

    moments = {
        "mu_mean": mu.mean().item(),
        "mu_var": mu.var().item(),
        "var_mean": var.mean().item(),
        "var_var": var.var().item(),
        "n_samples": args.n_samples,
    }

    # Save
    save_path = os.path.join(args.output_dir, "teacher_moments.pt")
    th.save(moments, save_path)

    print(f"\nTeacher moments saved to {save_path}")
    print(f"  mean(spatial_mean) = {moments['mu_mean']:.6f}")
    print(f"  var(spatial_mean)  = {moments['mu_var']:.6f}")
    print(f"  mean(spatial_var)  = {moments['var_mean']:.6f}")
    print(f"  var(spatial_var)   = {moments['var_var']:.6f}")


if __name__ == "__main__":
    main()
