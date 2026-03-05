"""
Generate samples from a Progressive Distillation student.

The student is a VPDiffusionModel that uses N DDIM steps (where N is the
number of steps it was distilled to). Sampling is just standard DDIM.

Usage:
    python scripts/sample_pd.py config/unet_pd_sample.yaml
"""

import sys
import os
import torch as th
import numpy as np
from omegaconf import OmegaConf

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.diffusion_utils import ddim_step


def ddim_sample(model, z, num_steps, schedule_s=0.008):
    """
    Standard DDIM sampling loop.

    Starting from z ~ N(0, I) at t=1, iteratively denoise using
    num_steps evenly spaced DDIM steps down to t=0.
    """
    # Time discretization: t_N = 1, t_{N-1}, ..., t_0 = 0
    times = th.linspace(1.0, 0.0, num_steps + 1, device=z.device)

    x = z
    for i in range(num_steps):
        t = times[i]
        s = times[i + 1]

        B = x.shape[0]
        t_batch = th.full((B,), t.item(), device=x.device)
        s_batch = th.full((B,), s.item(), device=x.device)

        # Clamp to avoid schedule singularities
        t_batch = t_batch.clamp(1e-4, 1 - 1e-4)
        s_batch = s_batch.clamp(0, 1 - 1e-4)

        with th.no_grad():
            x_hat = model.predict_x(x, t_batch, use_ema=True)
            x = ddim_step(x_hat, x, t_batch, s_batch, schedule_s)

    return x


def main(config_path):
    config = OmegaConf.load(config_path)

    th.manual_seed(config.get("th_seed", 0))
    np.random.seed(config.get("np_seed", 0))

    dev = th.device(config.get("device", "cuda"))

    # --- Build network ---
    network = UNetModel(
        dim=config.unet.dim,
        channel_mult=config.unet.channel_mult,
        num_channels=config.unet.num_channels,
        num_res_blocks=config.unet.num_res_blocks,
        num_head_channels=config.unet.num_head_channels,
        attention_resolutions=config.unet.attention_resolutions,
        dropout=config.unet.dropout,
        use_new_attention_order=config.unet.use_new_attention_order,
        use_scale_shift_norm=config.unet.use_scale_shift_norm,
        class_cond=config.unet.get("class_cond", False),
        num_classes=config.unet.get("num_classes", None),
    )

    schedule_s = config.pd.schedule_s
    student_steps = config.pd.student_steps

    model = VPDiffusionModel(
        network=network, schedule_s=schedule_s, infer=True,
    )

    # --- Load checkpoint ---
    ckpt_path = config.pd.checkpoint_path
    assert os.path.exists(ckpt_path), f"No checkpoint at {ckpt_path}"
    state = th.load(ckpt_path, map_location="cpu", weights_only=True)

    model.network.load_state_dict(state["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path}")

    model.to(dev)
    model.eval()

    # --- Generate ---
    total_samples = config.inference.total_samples_to_generate
    batch_size = config.inference.batch_size
    shape_str = config.inference.shape_of_sample
    shape = tuple(int(s) for s in shape_str.split(","))

    rounds = (total_samples + batch_size - 1) // batch_size
    generated = []

    print(f"Generating {total_samples} samples with {student_steps} DDIM steps...")

    for i in range(rounds):
        n = min(batch_size, total_samples - i * batch_size)
        z = th.randn(n, *shape, device=dev)

        samples = ddim_sample(model, z, student_steps, schedule_s)
        generated.append(samples.detach().cpu())
        print(f"  Batch {i+1}/{rounds} done")

    generated = th.cat(generated, dim=0)[:total_samples]

    # --- Denormalize if stats are available ---
    norm_stats_dir = config.inference.get("norm_stats_dir", None)
    if norm_stats_dir is not None:
        min_path = os.path.join(norm_stats_dir, "data_min.npy")
        max_path = os.path.join(norm_stats_dir, "data_max.npy")
        if os.path.exists(min_path) and os.path.exists(max_path):
            data_min = float(np.load(min_path))
            data_max = float(np.load(max_path))
            generated_denorm = (generated + 1.0) / 2.0 * (data_max - data_min) + data_min
            print(f"Denormalized range: [{generated_denorm.min():.4f}, {generated_denorm.max():.4f}]")

    # --- Save ---
    save_path = config.inference.save_path
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    th.save(generated, save_path)
    print(f"Saved {total_samples} samples to {save_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/sample_pd.py <path_to_config>")
        sys.exit(1)
    main(sys.argv[1])
