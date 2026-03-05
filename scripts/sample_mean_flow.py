"""
Generate samples from a trained Mean Flow Matching model.

Usage:
    python scripts/sample_mean_flow.py config/unet_mean_flow_sample.yaml
"""

import sys
import os
import torch as th
import numpy as np
from omegaconf import OmegaConf

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.flow_models import MeanFlowMatching
from src.inference.samplers import MeanSampler


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
        use_future_time_emb=config.unet.get("use_future_time_emb", False),
    )

    model = MeanFlowMatching(
        network=network,
        infer=True,
    )

    # --- Load checkpoint ---
    ckpt_path = config.mean_flow.checkpoint_path
    assert os.path.exists(ckpt_path), f"No checkpoint at {ckpt_path}"
    state = th.load(ckpt_path, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path}")

    model.to(dev)
    model.eval()

    # --- Sampler ---
    sampler = MeanSampler(model)

    # --- Generate ---
    total_samples = config.inference.total_samples_to_generate
    batch_size = config.inference.batch_size
    shape_str = config.inference.shape_of_sample
    shape = tuple(int(s) for s in shape_str.split(","))

    solver_kwargs = OmegaConf.to_container(
        config.inference.get("solver_kwargs", OmegaConf.create({})),
        resolve=True,
    )

    rounds = (total_samples + batch_size - 1) // batch_size
    generated = []

    t_span_info = solver_kwargs.get("t_span_kwargs", {"start": 0, "end": 1, "steps": 2})
    num_steps = t_span_info.get("steps", 2) - 1
    print(f"Generating {total_samples} samples with {num_steps} Euler step(s)...")

    for i in range(rounds):
        n = min(batch_size, total_samples - i * batch_size)
        z = th.randn(n, *shape, device=dev)

        with th.no_grad():
            samples = sampler.sample(z, **solver_kwargs)

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
        print("Usage: python scripts/sample_mean_flow.py <path_to_config>")
        sys.exit(1)
    main(sys.argv[1])
