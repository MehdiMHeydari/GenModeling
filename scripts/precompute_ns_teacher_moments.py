"""
Pre-compute NS teacher distribution moments from DDIM samples.

Same logic as precompute_teacher_moments.py but for the 2-channel NS teacher.
Settings hardcoded for NS: 2-channel UNet input, 250 DDIM steps, seed 0.

Usage:
    python scripts/precompute_ns_teacher_moments.py --gpu 5 --n_samples 1000 \\
        --checkpoint ns_teacher/exp_1/saved_state/checkpoint_75.pt
"""

import argparse
import os
import torch as th
import numpy as np

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.diffusion_utils import ddim_step
from src.eval.constants import TEACHER_DDIM_STEPS, SCHEDULE_S, SINGLE_SEED


NS_DATA_SHAPE = (2, 128, 128)

UNET_CFG = dict(
    dim=list(NS_DATA_SHAPE),
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to NS teacher checkpoint to sample from")
    p.add_argument("--output_dir", type=str,
                   default="ns_teacher/exp_1/saved_state")
    p.add_argument("--output_name", type=str, default="teacher_moments.pt")
    args = p.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")

    # Load teacher
    network = UNetModel(**UNET_CFG)
    teacher = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(args.checkpoint, map_location="cpu", weights_only=True)
    teacher.network.load_state_dict(state["model_state_dict"])
    teacher.to(device).eval()
    print(f"Loaded teacher from {args.checkpoint}")

    # Sample
    th.manual_seed(SINGLE_SEED)
    ts = th.linspace(1.0, 0.0, TEACHER_DDIM_STEPS + 1, device=device)
    batch_size = 32  # smaller batch for 2-channel
    all_samples = []

    print(f"Sampling {args.n_samples} from NS teacher "
          f"({TEACHER_DDIM_STEPS} DDIM steps, seed={SINGLE_SEED})...")
    with th.no_grad():
        for i in range(0, args.n_samples, batch_size):
            n = min(batch_size, args.n_samples - i)
            z = th.randn(n, *NS_DATA_SHAPE, device=device)
            for step in range(TEACHER_DDIM_STEPS):
                t_batch = th.full((n,), ts[step].item(), device=device)
                s_batch = th.full((n,), ts[step + 1].item(), device=device)
                x_hat = teacher.predict_x(z, t_batch)
                z = ddim_step(x_hat, z, t_batch, s_batch, SCHEDULE_S)
            all_samples.append(z.cpu())
            print(f"  {i + n}/{args.n_samples}")

    samples = th.cat(all_samples, dim=0)

    # Compute moments per-sample (flatten across channels + spatial)
    flat = samples.flatten(1)
    mu = flat.mean(dim=1)    # per-sample field mean
    var = flat.var(dim=1)     # per-sample field variance

    moments = {
        "mu_mean": mu.mean().item(),
        "mu_var": mu.var().item(),
        "var_mean": var.mean().item(),
        "var_var": var.var().item(),
        "n_samples": args.n_samples,
        "source_checkpoint": args.checkpoint,
    }

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_name)
    th.save(moments, save_path)

    print(f"\nTeacher moments saved to {save_path}")
    print(f"  mean(field_mean) = {moments['mu_mean']:.6f}")
    print(f"  var(field_mean)  = {moments['mu_var']:.6f}")
    print(f"  mean(field_var)  = {moments['var_mean']:.6f}")
    print(f"  var(field_var)   = {moments['var_var']:.6f}")


if __name__ == "__main__":
    main()
