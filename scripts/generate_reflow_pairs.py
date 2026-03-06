"""
Generate coupled (noise, data) pairs for Reflow training.

Takes a trained round-1 Rectified Flow model, samples random noise,
runs the ODE forward to get generated data, and saves (z, x) pairs.
These pairs are used for round-2 (reflow) training to straighten trajectories.

Usage:
    python scripts/generate_reflow_pairs.py \
        --checkpoint darcy_rectified_flow/exp_1/saved_state/checkpoint_799.pt \
        --n_pairs 9000 --ode_steps 100 \
        --save_path darcy_rectified_flow/reflow_pairs.pt
"""

import argparse
import os
import torch as th

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.flow_models import RectifiedFlowMatching
from src.inference.samplers import RectifiedFlowSampler


DATA_SHAPE = (1, 128, 128)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained round-1 checkpoint")
    parser.add_argument("--n_pairs", type=int, default=9000,
                        help="Number of (z, x) pairs to generate (match training set size)")
    parser.add_argument("--ode_steps", type=int, default=100,
                        help="Number of Euler steps for ODE integration (more = more accurate)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_path", type=str, default="darcy_rectified_flow/reflow_pairs.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    th.manual_seed(args.seed)

    # --- Build and load model ---
    network = UNetModel(**UNET_CFG)
    model = RectifiedFlowMatching(network=network, add_heavy_noise=False, infer=True)

    state = th.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded: {args.checkpoint}")

    # --- Generate pairs ---
    sampler = RectifiedFlowSampler(model)
    n_total = args.n_pairs
    batch_size = args.batch_size
    rounds = (n_total + batch_size - 1) // batch_size

    all_z = []
    all_x = []

    print(f"Generating {n_total} coupled pairs with {args.ode_steps} ODE steps...")

    for i in range(rounds):
        n = min(batch_size, n_total - i * batch_size)
        z = th.randn(n, *DATA_SHAPE, device=device)
        x = sampler.sample(z, num_steps=args.ode_steps)

        all_z.append(z.cpu())
        all_x.append(x.cpu())

        if (i + 1) % 10 == 0 or i == rounds - 1:
            print(f"  Batch {i+1}/{rounds}")

    all_z = th.cat(all_z, dim=0)[:n_total]
    all_x = th.cat(all_x, dim=0)[:n_total]

    # --- Save ---
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    th.save({"z": all_z, "x": all_x}, args.save_path)
    print(f"Saved {n_total} pairs to {args.save_path}")
    print(f"  z shape: {all_z.shape}, range: [{all_z.min():.3f}, {all_z.max():.3f}]")
    print(f"  x shape: {all_x.shape}, range: [{all_x.min():.3f}, {all_x.max():.3f}]")


if __name__ == "__main__":
    main()
