"""
Pre-compute teacher distribution moments from DDIM samples and save to disk.
These are used as targets for the sampling-based moment loss during CD training.

Settings are pulled from `src.eval.constants` so teacher sampling is locked
to the canonical paper config for the chosen dataset (via --dataset; the
canonical ckpt and DDIM step count come from `_DATASETS`).

Usage:
    python scripts/precompute_teacher_moments.py --gpu 0 --n_samples 1000
    python scripts/precompute_teacher_moments.py --gpu 0 --dataset cns
"""

import argparse
import os
import torch as th
import numpy as np

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.diffusion_utils import ddim_step
from src.eval.constants import SCHEDULE_S, SINGLE_SEED, get_dataset


def make_unet_cfg(data_shape):
    return dict(
        dim=list(data_shape),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--dataset", type=str, default="darcy")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Defaults to the dataset's stats_dir")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite teacher_moments.pt instead of "
                             "writing teacher_moments_v2.pt when it exists")
    args = parser.parse_args()

    ds = get_dataset(args.dataset)
    teacher_ckpt = ds["teacher_ckpt"]
    ddim_steps = ds["teacher_ddim_steps"]
    data_shape = ds["data_shape"]
    output_dir = args.output_dir or ds["stats_dir"]
    eta = float(ds.get("teacher_eta", 0.0))

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")

    # Load teacher
    network = UNetModel(**make_unet_cfg(data_shape))
    teacher = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(teacher_ckpt, map_location="cpu", weights_only=True)
    teacher.network.load_state_dict(state["model_state_dict"])
    teacher.to(device).eval()
    print(f"dataset={args.dataset} teacher={teacher_ckpt} steps={ddim_steps}")

    # Sample (canonical: SINGLE_SEED + the dataset's canonical DDIM steps)
    th.manual_seed(SINGLE_SEED)
    ts = th.linspace(1.0, 0.0, ddim_steps + 1, device=device)
    batch_size = 64
    all_samples = []

    print(f"Sampling {args.n_samples} from teacher "
          f"({ddim_steps} DDIM steps, seed={SINGLE_SEED})...")
    with th.no_grad():
        for i in range(0, args.n_samples, batch_size):
            n = min(batch_size, args.n_samples - i)
            z = th.randn(n, *data_shape, device=device)
            for step in range(ddim_steps):
                t_batch = th.full((n,), ts[step].item(), device=device)
                s_batch = th.full((n,), ts[step + 1].item(), device=device)
                x_hat = teacher.predict_x(z, t_batch)
                # canonical protocol may be stochastic (teacher_eta in
                # _DATASETS); never inject noise on the final step
                if eta > 0.0 and step < ddim_steps - 1:
                    from scripts.evaluate_paper import ddim_eta_step
                    z = ddim_eta_step(x_hat, z, t_batch, s_batch, SCHEDULE_S, eta)
                else:
                    z = ddim_step(x_hat, z, t_batch, s_batch, SCHEDULE_S)
            all_samples.append(z.cpu())
            print(f"  {i + n}/{args.n_samples}")

    samples = th.cat(all_samples, dim=0)

    # Compute moments (pooled across channels — the original formulation)
    flat = samples.flatten(1)
    mu = flat.mean(dim=1)    # per-sample spatial mean
    var = flat.var(dim=1)     # per-sample spatial variance

    moments = {
        "mu_mean": mu.mean().item(),
        "mu_var": mu.var().item(),
        "var_mean": var.mean().item(),
        "var_var": var.var().item(),
        "n_samples": args.n_samples,
        "ddim_steps": ddim_steps,
        "eta": eta,
    }

    # Per-channel targets (for moment_per_channel PRISM): same statistics
    # computed independently for each channel.
    B, C = samples.shape[0], samples.shape[1]
    flat_ch = samples.reshape(B, C, -1)
    mu_c = flat_ch.mean(dim=2)   # (B, C)
    var_c = flat_ch.var(dim=2)   # (B, C)
    moments["mu_mean_ch"] = mu_c.mean(dim=0)
    moments["mu_var_ch"] = mu_c.var(dim=0)
    moments["var_mean_ch"] = var_c.mean(dim=0)
    moments["var_var_ch"] = var_c.var(dim=0)

    # Spectral-band targets (for moment_spectral PRISM): per-sample,
    # per-channel log mean power in radial wavenumber bands (DC excluded),
    # summarized as across-sample mean and variance.
    band_edges = (1, 4, 16, 48, 64)
    H, W = samples.shape[-2:]
    cy, cx = H // 2, W // 2
    yy, xx = th.meshgrid(th.arange(H), th.arange(W), indexing="ij")
    r = th.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
    P = th.fft.fftshift(th.fft.fft2(samples).abs() ** 2, dim=(-2, -1))
    stats = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        m = ((r >= lo) & (r < hi)).float()
        stats.append(th.log((P * m).sum(dim=(-2, -1)) / m.sum().clamp(min=1.0) + 1e-12))
    spec = th.stack(stats, dim=-1)          # (B, C, K)
    moments["spec_mean_ch"] = spec.mean(dim=0)
    moments["spec_var_ch"] = spec.var(dim=0)
    moments["spec_band_edges"] = th.tensor(band_edges)

    # Save (never clobber a file an existing run trained against: bump
    # the version suffix until a free name is found)
    save_path = os.path.join(output_dir, "teacher_moments.pt")
    if os.path.exists(save_path) and not args.overwrite:
        v = 2
        while os.path.exists(os.path.join(output_dir, f"teacher_moments_v{v}.pt")):
            v += 1
        save_path = os.path.join(output_dir, f"teacher_moments_v{v}.pt")
        print(f"Existing moment files kept; writing {save_path}")
    th.save(moments, save_path)

    print(f"\nTeacher moments saved to {save_path}")
    print(f"  mean(spatial_mean) = {moments['mu_mean']:.6f}")
    print(f"  var(spatial_mean)  = {moments['mu_var']:.6f}")
    print(f"  mean(spatial_var)  = {moments['var_mean']:.6f}")
    print(f"  var(spatial_var)   = {moments['var_var']:.6f}")


if __name__ == "__main__":
    main()
