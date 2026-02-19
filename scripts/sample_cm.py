"""
Generate samples from a trained Multistep Consistency Model.

Usage:
    python scripts/sample_cm.py config/unet_cm_sample.yaml
"""

import sys
import os
import torch as th
import numpy as np
from omegaconf import OmegaConf

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.consistency_models import MultistepConsistencyModel
from src.inference.samplers import MultistepCMSampler


def main(config_path):
    config = OmegaConf.load(config_path)

    th.manual_seed(config.get("th_seed", 0))
    np.random.seed(config.get("np_seed", 0))

    dev = th.device(config.get("device", "cuda"))

    # --- Build network ---
    class_cond = config.unet.get("class_cond", False)

    network = UNetModel(
        dim=config.unet.dim,
        channel_mult=config.unet.channel_mult,
        num_channels=config.unet.num_channels,
        num_res_blocks=config.unet.res_blocks,
        num_head_channels=config.unet.head_chans,
        attention_resolutions=config.unet.attn_res,
        dropout=config.unet.dropout,
        use_new_attention_order=config.unet.new_attn,
        use_scale_shift_norm=config.unet.film,
        class_cond=class_cond,
        num_classes=config.unet.get("num_classes", None),
    )

    schedule_s = config.get("schedule_s", 0.008)
    student_steps = config.cd.student_steps

    model = MultistepConsistencyModel(
        network=network,
        student_steps=student_steps,
        schedule_s=schedule_s,
        infer=True,
    )

    # --- Load checkpoint ---
    ckpt_path = config.cd.checkpoint_path
    assert os.path.exists(ckpt_path), f"No checkpoint at {ckpt_path}"
    state = th.load(ckpt_path, map_location="cpu", weights_only=True)

    model.network.load_state_dict(state["model_state_dict"])
    if "ema_state_dict" in state:
        model.ema_network.load_state_dict(state["ema_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path}")

    model.to(dev)
    model.eval()

    # --- Sampler ---
    use_ema = config.inference.get("use_ema", True)
    sampler = MultistepCMSampler(model)

    # --- Generate ---
    total_samples = config.inference.total_samples_to_generate
    batch_size = config.inference.batch_size
    shape_str = config.inference.shape_of_sample
    shape = tuple(int(s) for s in shape_str.split(","))

    rounds = (total_samples + batch_size - 1) // batch_size
    generated = []

    print(f"Generating {total_samples} samples with {student_steps} steps...")

    for i in range(rounds):
        n = min(batch_size, total_samples - i * batch_size)
        z = th.randn(n, *shape, device=dev)

        # Class labels if needed
        kwargs = {}
        if class_cond and config.inference.get("class_label") is not None:
            y = th.full((n,), config.inference.class_label, device=dev, dtype=th.long)
            kwargs["y"] = y

        with th.no_grad():
            samples = sampler.sample(z, **kwargs)

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
        print("Usage: python scripts/sample_cm.py <path_to_config>")
        sys.exit(1)
    main(sys.argv[1])
