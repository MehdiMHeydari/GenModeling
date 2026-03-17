"""
Train a Multistep Consistency Model via Consistency Distillation (CD).

Requires a pretrained VP diffusion model checkpoint as the teacher.
The student is initialized from the teacher weights.

Usage:
    python scripts/train_cm.py config/unet_cm_cd.yaml
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
from src.training.objectives import MultistepCDLoss
from src.models.vp_diffusion import VPDiffusionModel
from src.models.consistency_models import MultistepConsistencyModel


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
        'ema_state_dict': model.ema_network.state_dict(),
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

    schedule_s = config.cd.get("schedule_s", 0.008)

    # --- Teacher model (frozen) ---
    teacher_network = build_unet(config)
    teacher = VPDiffusionModel(
        network=teacher_network, schedule_s=schedule_s, infer=True,
    )
    teacher_state = th.load(
        config.cd.teacher_checkpoint, map_location='cpu', weights_only=True,
    )
    # Prefer raw weights (empirically better than EMA for this training)
    teacher.network.load_state_dict(teacher_state['model_state_dict'])
    print("Using teacher raw weights")
    teacher.to(dev)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"Loaded teacher from {config.cd.teacher_checkpoint}")

    # --- Student consistency model (init from teacher) ---
    student_network = build_unet(config)
    if config.cd.get("init_from_teacher", True):
        student_network.load_state_dict(teacher_network.state_dict())
        print("Initialized student from teacher weights")

    model = MultistepConsistencyModel(
        network=student_network,
        student_steps=config.cd.student_steps,
        schedule_s=schedule_s,
        ema_rate=config.cd.get("ema_rate", 0.9999),
    )
    model.to(dev)

    # --- Loss ---
    objective = MultistepCDLoss(
        class_conditional=config.unet.get("class_cond", False),
        teacher_model=teacher,
        student_steps=config.cd.student_steps,
        x_var_frac=config.cd.get("x_var_frac", 0.75),
        huber_epsilon=config.cd.get("huber_epsilon", 1e-4),
        schedule_s=schedule_s,
        moment_weight_mu=config.cd.get("moment_weight_mu", 0.0),
        moment_weight_var=config.cd.get("moment_weight_var", 0.0),
        teacher_moments_path=config.cd.get("teacher_moments_path", None),
        moment_every=config.cd.get("moment_every", 50),
        moment_batch_size=config.cd.get("moment_batch_size", 32),
    )

    # --- Optimizer ---
    optim = Adam(model.network.parameters(), lr=config.optimizer.lr)

    # --- Weights & Biases ---
    wandb.init(
        project="darcy-student",
        name=f"cd_exp{config.exp_num}",
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
        if 'ema_state_dict' in state:
            model.ema_network.load_state_dict(state['ema_state_dict'])
        optim.load_state_dict(state['optimizer_state_dict'])
        start_epoch = restart_epoch + 1
        print(f"Resumed from epoch {restart_epoch}")

    # --- Training loop ---
    num_epochs = config.train.num_epochs
    save_interval = config.train.checkpointing_dict.save_epoch_int
    grad_accum = config.train.get("grad_accum_steps", 1)
    best_loss = float('inf')

    print(f"Starting CD training: {num_epochs} epochs, "
          f"student_steps={config.cd.student_steps}, "
          f"grad_accum={grad_accum} (effective batch={config.dataloader.batch_size * grad_accum})")

    for epoch in tqdm(range(start_epoch, num_epochs), desc="CD Student"):
        model.network.train()
        total_loss = 0.0

        optim.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            loss = objective(model, batch, device=dev) / grad_accum
            loss.backward()

            # Sampling-based moment loss (separate backward, after CD graph is freed)
            moment_loss = objective.sample_moment_loss(model, dev)
            if moment_loss is not None:
                moment_loss.backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                optim.step()
                model.update_ema()
                optim.zero_grad()

            total_loss += loss.item() * grad_accum

        avg_loss = total_loss / len(train_loader)
        best_loss = min(best_loss, avg_loss)
        n_teacher = objective._teacher_step_schedule()

        log_dict = {"loss": avg_loss, "best_loss": best_loss,
                    "N_teacher": n_teacher, "epoch": epoch}
        if config.cd.get("moment_weight_mu", 0) > 0 or config.cd.get("moment_weight_var", 0) > 0:
            log_dict["moment_loss_mu"] = objective.last_moment_mu
            log_dict["moment_loss_var"] = objective.last_moment_var
        wandb.log(log_dict)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            tqdm.write(f"Epoch {epoch}: loss={avg_loss:.6f}, "
                       f"best={best_loss:.6f}, N_teacher={n_teacher}")

        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            save_checkpoint(model, optim, epoch, savepath)
            tqdm.write(f"  Saved checkpoint_{epoch}.pt")

    # Save final
    save_checkpoint(model, optim, num_epochs - 1, savepath)
    print(f"CD training complete. Best loss: {best_loss:.6f}")
    print(f"Checkpoints in {savepath}")
    wandb.finish()


if __name__ == '__main__':
    main(sys.argv[1])
