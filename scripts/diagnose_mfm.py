"""
Quick diagnostic for MFM training.
Checks: loss value, gradient norms, sampling with 1/2/4 steps.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/diagnose_mfm.py
"""

import os
import glob
import re
import torch as th
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.flow_models import MeanFlowMatching
from src.training.objectives import MeanFlowMatchingLoss
from src.inference.samplers import MeanSampler
from src.utils.dataloader import get_darcy_loader
from src.utils.dataset import DATASETS


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
    use_future_time_emb=True,
)


def find_latest_checkpoint(save_dir):
    ckpts = glob.glob(os.path.join(save_dir, "checkpoint_*.pt"))
    if not ckpts:
        return None, None
    def epoch_num(p):
        m = re.search(r"checkpoint_(\d+)\.pt", p)
        return int(m.group(1)) if m else -1
    best = max(ckpts, key=epoch_num)
    return best, epoch_num(best)


def main():
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    th.manual_seed(42)

    save_dir = "darcy_mean_flow/exp_2/saved_state"
    ckpt, epoch = find_latest_checkpoint(save_dir)
    if ckpt is None:
        print("No checkpoints found!")
        return
    print(f"Checkpoint: {ckpt} (epoch {epoch})")

    # --- Load model ---
    network = UNetModel(**UNET_CFG)
    model = MeanFlowMatching(network=network)
    state = th.load(ckpt, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device)

    # --- Load a batch of real data ---
    train_loader, data_min, data_max = get_darcy_loader(
        data_path="data/2D_DarcyFlow_beta1.0_Train.hdf5",
        batch_size=16, dataset_cls=DATASETS["VF_FM"], train_samples=9000,
    )
    batch = next(iter(train_loader))
    x0, x1 = batch
    x0, x1 = x0.to(device), x1.to(device)

    # --- Check training loss and gradient norms ---
    print("\n=== TRAINING DIAGNOSTICS ===")
    model.network.train()

    # Forward pass
    ut_pred, ut = model(x0, x1, None)
    delta = ut_pred - ut
    delta_l2_sq = delta.view(delta.shape[0], -1).pow(2).sum(dim=1)

    print(f"  ||u_pred||  mean: {ut_pred.view(ut_pred.shape[0], -1).norm(dim=1).mean():.4f}")
    print(f"  ||u_target|| mean: {ut.view(ut.shape[0], -1).norm(dim=1).mean():.4f}")
    print(f"  ||delta||^2  mean: {delta_l2_sq.mean():.4f}, min: {delta_l2_sq.min():.4f}, max: {delta_l2_sq.max():.4f}")

    # Adaptive weight
    gamma = 0.0
    w = (1. / (delta_l2_sq + 1e-3) ** (1 - gamma)).detach()
    print(f"  Adaptive w   mean: {w.mean():.6f}, min: {w.min():.6f}, max: {w.max():.6f}")

    # Loss with gamma=0 (adaptive)
    loss_adaptive = (w * delta_l2_sq).mean()
    print(f"  Loss (gamma=0, adaptive): {loss_adaptive.item():.6f}")

    # Loss with gamma=1 (plain MSE)
    loss_mse = delta_l2_sq.mean()
    print(f"  Loss (gamma=1, plain MSE): {loss_mse.item():.4f}")

    # Gradient norm check
    model.network.zero_grad()
    loss_adaptive.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.network.parameters() if p.grad is not None) ** 0.5
    print(f"  Gradient norm (adaptive): {grad_norm:.6f}")

    # --- Sampling with different step counts ---
    print("\n=== SAMPLING DIAGNOSTICS ===")
    model.infer = True          # switch dispatch to sample(r, t, xt)
    model.network.eval()
    sampler = MeanSampler(model)

    # Shared noise
    z = th.randn(8, *DATA_SHAPE, device=device)

    fig, axes = plt.subplots(4, 8, figsize=(20, 10))

    # Row 0: Real data
    real = x1[:8].cpu().numpy()
    for j in range(8):
        axes[0, j].imshow(real[j, 0], cmap="RdBu_r")
        axes[0, j].axis("off")
    axes[0, 0].set_ylabel("Real Data", fontsize=11, fontweight="bold",
                           rotation=0, labelpad=80, va="center")

    # Rows 1-3: MFM with 1, 2, 4 steps
    for row, n_steps in enumerate([1, 2, 4], start=1):
        t_span_kwargs = {"start": 0, "end": 1, "steps": n_steps + 1}
        with th.no_grad():
            samples = sampler.sample(z, t_span_kwargs=t_span_kwargs)

        # Denormalize
        samples_np = samples.cpu().numpy()
        samples_denorm = (samples_np + 1.0) / 2.0 * (data_max - data_min) + data_min

        for j in range(8):
            axes[row, j].imshow(samples_denorm[j, 0], cmap="RdBu_r")
            axes[row, j].axis("off")
        axes[row, 0].set_ylabel(f"MFM\n({n_steps} step{'s' if n_steps > 1 else ''})",
                                 fontsize=11, fontweight="bold",
                                 rotation=0, labelpad=80, va="center")

        print(f"  {n_steps}-step samples: range [{samples.min():.3f}, {samples.max():.3f}], "
              f"std={samples.std():.3f}")

    for j in range(8):
        axes[0, j].set_title(f"Sample {j+1}", fontsize=10, fontweight="bold")

    fig.suptitle(f"MFM Diagnostic (epoch {epoch})", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0.08, 0, 1, 0.95])
    plt.savefig("mfm_diagnostic.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved mfm_diagnostic.png")


if __name__ == "__main__":
    main()
