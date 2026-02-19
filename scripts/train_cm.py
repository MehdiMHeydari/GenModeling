"""
Train a Multistep Consistency Model via Consistency Distillation (CD).

Requires a pretrained VP diffusion model checkpoint as the teacher.
The student is optionally initialized from the teacher weights.

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

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.utils.dataloader import get_loaders_vf_fm
from src.utils.dataset import DATASETS
from src.training.objectives import MultistepCDLoss
from src.models.vp_diffusion import VPDiffusionModel
from src.models.consistency_models import MultistepConsistencyModel
from src.utils.logger import configure, logkvs, log, dumpkvs


def create_dir(path, config):
    if os.path.exists(path):
        if not config.train.checkpointing_dict.restart:
            print(f"Directory '{path}' already exists and restart is False. Deleting and recreating directory.")
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            print(f"Directory '{path}' already exists. Resuming training.")
    else:
        os.makedirs(path)
        print(f"Directory '{path}' created.")


def save_checkpoint(model, optimizer, epoch, save_path):
    """Save model, EMA, and optimizer state."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.network.state_dict(),
        'ema_state_dict': model.ema_network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    th.save(state, f"{save_path}/checkpoint_{epoch}.pt")


def main(config_path):
    config = OmegaConf.load(config_path)

    th.manual_seed(config.th_seed)
    np.random.seed(config.np_seed)

    logpath = config.path + f"/exp_{config.exp_num}"
    savepath = logpath + "/saved_state"
    create_dir(logpath, config=config)
    create_dir(savepath, config=config)

    configure(
        dir=logpath,
        format_strs=config.train.logger_dict.format_strs,
        config=config,
    )

    dev = th.device(config.device)

    # --- Data ---
    train_dataloader = get_loaders_vf_fm(
        vf_paths=config.dataloader.datapath,
        batch_size=config.dataloader.batch_size,
        dataset_=DATASETS[config.dataloader.dataset],
    )

    class_cond = config.unet.class_cond if hasattr(config.unet, 'class_cond') else False

    # --- Teacher model (frozen) ---
    teacher_network = UNetModel(
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
        num_classes=config.unet.num_classes if hasattr(config.unet, 'num_classes') else None,
    )

    schedule_s = config.cd.get("schedule_s", 0.008)
    teacher = VPDiffusionModel(network=teacher_network, schedule_s=schedule_s, infer=True)

    teacher_state = th.load(config.cd.teacher_checkpoint, map_location='cpu', weights_only=True)
    teacher.network.load_state_dict(teacher_state['model_state_dict'])
    teacher.to(dev)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    log(f"Loaded teacher from {config.cd.teacher_checkpoint}")

    # --- Student consistency model ---
    student_network = UNetModel(
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
        num_classes=config.unet.num_classes if hasattr(config.unet, 'num_classes') else None,
    )

    # Optionally initialize student from teacher weights
    if config.cd.get("init_from_teacher", True):
        student_network.load_state_dict(teacher_network.state_dict())
        log("Initialized student from teacher weights")

    model = MultistepConsistencyModel(
        network=student_network,
        student_steps=config.cd.student_steps,
        schedule_s=schedule_s,
        ema_rate=config.cd.get("ema_rate", 0.9999),
    )
    model.to(dev)

    # --- Loss ---
    objective = MultistepCDLoss(
        class_conditional=class_cond,
        teacher_model=teacher,
        student_steps=config.cd.student_steps,
        x_var_frac=config.cd.get("x_var_frac", 0.75),
        huber_epsilon=config.cd.get("huber_epsilon", 1e-4),
        schedule_s=schedule_s,
    )

    # --- Optimizer ---
    optim = Adam(model.network.parameters(), lr=config.optimizer.lr)

    # --- Resume if requested ---
    start_epoch = 0
    if config.train.checkpointing_dict.get("restart", False):
        restart_epoch = config.train.checkpointing_dict.restart_epoch
        ckpt_path = f"{savepath}/checkpoint_{restart_epoch}.pt"
        assert os.path.exists(ckpt_path), f"No checkpoint at {ckpt_path}"
        state = th.load(ckpt_path, map_location='cpu', weights_only=True)
        model.network.load_state_dict(state['model_state_dict'])
        if 'ema_state_dict' in state:
            model.ema_network.load_state_dict(state['ema_state_dict'])
        optim.load_state_dict(state['optimizer_state_dict'])
        start_epoch = restart_epoch + 1
        log(f"Resumed from epoch {restart_epoch}")

    # --- Training loop ---
    num_epochs = config.train.num_epochs
    save_epoch_int = config.train.checkpointing_dict.save_epoch_int
    log_batch_int = config.train.checkpointing_dict.log_batch_int
    log_print_freq = config.train.logger_dict.log_print_freq

    log(f"Starting Multistep CD training: {num_epochs} epochs, "
        f"student_steps={config.cd.student_steps}")

    for epoch in range(start_epoch, num_epochs):
        log(f"Starting epoch {epoch}")
        train_loss = 0.0
        num_batches = 0

        model.network.train()

        for batch in train_dataloader:
            loss = objective(model, batch, device=dev)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # EMA update after each gradient step
            model.update_ema()

            loss_val = loss.item()
            train_loss += loss_val
            num_batches += 1

            if num_batches % log_batch_int == 0:
                logkvs({"batch": num_batches, "Minibatch loss": loss_val,
                        "teacher_steps": objective._teacher_step_schedule(),
                        "iteration": objective._iteration})
                log(f"  Epoch {epoch}, Batch {num_batches}: loss={loss_val:.6f}")
                dumpkvs()

        train_loss = train_loss / max(num_batches, 1)
        logkvs({"epoch": epoch, "Epoch loss": train_loss})

        if epoch % log_print_freq == 0:
            log(f"Average Training loss at epoch {epoch}: {train_loss:.6f}")

        if epoch % save_epoch_int == 0:
            log(f"Saving state at epoch {epoch}...")
            save_checkpoint(model, optim, epoch, savepath)

        dumpkvs()

    # Save final checkpoint
    save_checkpoint(model, optim, num_epochs - 1, savepath)
    log("Training complete")


if __name__ == '__main__':
    main(sys.argv[1])
