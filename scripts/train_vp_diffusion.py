"""
Train a VP diffusion model (teacher for consistency distillation).

Usage:
    python scripts/train_vp_diffusion.py config/unet_vp_diffusion.yaml
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
from src.utils.dataloader import get_data_loader
from src.utils.dataset import DATASETS
from src.models.vp_diffusion import VPDiffusionModel
from src.training.objectives import VPDiffusionLoss


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
    # Save EMA weights if available
    if model.ema_network is not None:
        state['ema_state_dict'] = model.ema_network.state_dict()
    th.save(state, f"{save_path}/checkpoint_{epoch}.pt")


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

    # --- Data ---
    train_loader, data_min, data_max = get_data_loader(
        data_path=config.dataloader.datapath,
        batch_size=config.dataloader.batch_size,
        dataset_cls=DATASETS[config.dataloader.dataset],
        train_samples=config.dataloader.get("train_samples", 9000),
        save_dir=savepath,
        loader_type=config.dataloader.get("loader_type", "darcy"),
    )
    print(f"Data loaded: {len(train_loader)} batches/epoch, "
          f"range [{data_min:.4f}, {data_max:.4f}]")

    # --- Network ---
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
    total_params = sum(p.numel() for p in network.parameters())
    print(f"UNet parameters: {total_params:,}")

    schedule_s = config.get("vp", {}).get("schedule_s", 0.008)
    ema_rate = config.get("vp", {}).get("ema_rate", 0.9999)
    model = VPDiffusionModel(network=network, schedule_s=schedule_s, ema_rate=ema_rate)
    model.to(dev)
    print(f"EMA rate: {ema_rate}")

    # --- Optimizer ---
    num_epochs = config.train.num_epochs
    optim = Adam(model.network.parameters(), lr=config.optimizer.lr)

    # --- Weights & Biases ---
    wandb.init(
        project="darcy-teacher",
        name=f"vp_diffusion_exp{config.exp_num}",
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
        if 'ema_state_dict' in state and model.ema_network is not None:
            model.ema_network.load_state_dict(state['ema_state_dict'])
        optim.load_state_dict(state['optimizer_state_dict'])
        start_epoch = restart_epoch + 1
        print(f"Resumed from epoch {restart_epoch}")

    # --- Loss ---
    objective = VPDiffusionLoss(
        class_conditional=config.unet.get("class_cond", False)
    )

    # --- Training loop ---
    save_interval = config.train.checkpointing_dict.save_epoch_int
    best_loss = float('inf')

    print(f"Starting teacher training: {num_epochs} epochs, "
          f"batch_size={config.dataloader.batch_size}, lr={config.optimizer.lr}")

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Teacher"):
        model.network.train()
        total_loss = 0.0

        for batch in train_loader:
            loss = objective(model, batch, device=dev)
            optim.zero_grad()
            loss.backward()
            optim.step()
            model.update_ema()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        best_loss = min(best_loss, avg_loss)

        wandb.log({"loss": avg_loss, "best_loss": best_loss, "epoch": epoch})

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            tqdm.write(f"Epoch {epoch}: loss={avg_loss:.6f}, "
                       f"best={best_loss:.6f}")

        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            save_checkpoint(model, optim, epoch, savepath)
            tqdm.write(f"  Saved checkpoint_{epoch}.pt")

    # Save final
    save_checkpoint(model, optim, num_epochs - 1, savepath)
    print(f"Training complete. Best loss: {best_loss:.6f}")
    print(f"Checkpoints in {savepath}")
    wandb.finish()


if __name__ == '__main__':
    main(sys.argv[1])
