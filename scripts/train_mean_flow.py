"""
Train a Mean Flow Matching model.

No teacher model required — trains directly on data using JVP-based loss
with dual time variables (t, r).

Usage:
    python scripts/train_mean_flow.py config/unet_mean_flow.yaml
"""

import sys
from omegaconf import OmegaConf
import os
import shutil
import torch as th
import numpy as np
from torch.optim import Adam
from tqdm.auto import tqdm
import wandb

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.utils.dataloader import get_darcy_loader
from src.utils.dataset import DATASETS
from src.training.objectives import MeanFlowMatchingLoss
from src.models.flow_models import MeanFlowMatching


def create_dir(path, restart=False):
    if os.path.exists(path):
        if not restart:
            print(f"Directory '{path}' already exists. Deleting and recreating.")
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            print(f"Directory '{path}' already exists. Resuming.")
    else:
        os.makedirs(path)
        print(f"Directory '{path}' created.")


def save_checkpoint(model, optimizer, epoch, save_path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    th.save(state, f"{save_path}/checkpoint_{epoch}.pt")


def build_unet(config):
    return UNetModel(
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


def main(config_path):
    config = OmegaConf.load(config_path)

    th.manual_seed(config.th_seed)
    np.random.seed(config.np_seed)

    logpath = config.path + f"/exp_{config.exp_num}"
    savepath = logpath + "/saved_state"
    restart = config.train.checkpointing_dict.get("restart", False)
    create_dir(logpath, restart=restart)
    create_dir(savepath, restart=restart)

    dev = th.device(config.device)

    # --- Data (HDF5 Darcy Flow) ---
    train_loader, data_min, data_max = get_darcy_loader(
        data_path=config.dataloader.datapath,
        batch_size=config.dataloader.batch_size,
        dataset_cls=DATASETS[config.dataloader.dataset],
        train_samples=config.dataloader.get("train_samples", 9000),
        save_dir=savepath,
    )
    print(f"Data loaded: {len(train_loader)} batches/epoch, "
          f"range [{data_min:.4f}, {data_max:.4f}]")

    # --- Mean Flow Model ---
    network = build_unet(config)
    model = MeanFlowMatching(
        network=network,
        t_schedule=config.mean_flow.get("t_schedule", "uniform"),
        log_norm_args=config.mean_flow.get("log_norm_args", (-0.4, 1.0)),
    )
    model.to(dev)

    # --- Loss ---
    objective = MeanFlowMatchingLoss(
        class_conditional=config.unet.get("class_cond", False),
        gamma=config.mean_flow.get("gamma", 0.0),
    )

    # --- Optimizer ---
    optim = Adam(model.network.parameters(), lr=config.optimizer.lr)

    # --- Weights & Biases ---
    wandb.init(
        project="darcy-mean-flow",
        name=f"mfm_exp{config.exp_num}",
        config=OmegaConf.to_container(config, resolve=True),
    )

    # --- Resume ---
    start_epoch = 0
    if restart:
        restart_epoch = config.train.checkpointing_dict.restart_epoch
        ckpt_path = f"{savepath}/checkpoint_{restart_epoch}.pt"
        assert os.path.exists(ckpt_path), f"No checkpoint at {ckpt_path}"
        state = th.load(ckpt_path, map_location='cpu', weights_only=True)
        model.network.load_state_dict(state['model_state_dict'])
        optim.load_state_dict(state['optimizer_state_dict'])
        start_epoch = restart_epoch + 1
        print(f"Resumed from epoch {restart_epoch}")

    # --- Training loop ---
    num_epochs = config.train.num_epochs
    save_interval = config.train.checkpointing_dict.save_epoch_int
    grad_accum = config.train.get("grad_accum_steps", 1)
    best_loss = float('inf')

    print(f"Starting Mean Flow training: {num_epochs} epochs, "
          f"grad_accum={grad_accum} (effective batch={config.dataloader.batch_size * grad_accum})")

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Mean Flow"):
        model.network.train()
        total_loss = 0.0

        optim.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            loss = objective(model, batch, device=dev) / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                grad_norm = th.nn.utils.clip_grad_norm_(model.network.parameters(), max_norm=1.0)
                optim.step()
                optim.zero_grad()

            total_loss += loss.item() * grad_accum

        avg_loss = total_loss / len(train_loader)
        best_loss = min(best_loss, avg_loss)

        wandb.log({"loss": avg_loss, "best_loss": best_loss,
                    "grad_norm": grad_norm.item(), "epoch": epoch})

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            tqdm.write(f"Epoch {epoch}: loss={avg_loss:.6f}, "
                       f"best={best_loss:.6f}")

        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            save_checkpoint(model, optim, epoch, savepath)
            tqdm.write(f"  Saved checkpoint_{epoch}.pt")

    # Save final
    save_checkpoint(model, optim, num_epochs - 1, savepath)
    print(f"Mean Flow training complete. Best loss: {best_loss:.6f}")
    print(f"Checkpoints in {savepath}")
    wandb.finish()


if __name__ == '__main__':
    main(sys.argv[1])
