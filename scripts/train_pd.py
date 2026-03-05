"""
Train a student via Progressive Distillation (Salimans & Ho, ICLR 2022).

Iteratively halves DDIM sampling steps: each round trains a student to
match 2 teacher DDIM steps in 1 forward pass. After each round the student
becomes the new teacher and the number of steps is halved.

Usage:
    python scripts/train_pd.py config/unet_pd.yaml
"""

import sys
import math
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
from src.training.objectives import ProgressiveDistillationLoss
from src.models.vp_diffusion import VPDiffusionModel


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


def save_checkpoint(model, optimizer, round_num, student_steps, save_path):
    state = {
        'round': round_num,
        'student_steps': student_steps,
        'model_state_dict': model.network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    th.save(state, f"{save_path}/pd_round{round_num}_steps{student_steps}.pt")


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

    # --- Data ---
    train_loader, data_min, data_max = get_darcy_loader(
        data_path=config.dataloader.datapath,
        batch_size=config.dataloader.batch_size,
        dataset_cls=DATASETS[config.dataloader.dataset],
        train_samples=config.dataloader.get("train_samples", 9000),
        save_dir=savepath,
    )
    print(f"Data loaded: {len(train_loader)} batches/epoch, "
          f"range [{data_min:.4f}, {data_max:.4f}]")

    schedule_s = config.pd.get("schedule_s", 0.008)

    # --- Load initial teacher ---
    teacher_network = build_unet(config)
    teacher = VPDiffusionModel(
        network=teacher_network, schedule_s=schedule_s, infer=True,
    )
    ckpt_path = config.pd.teacher_checkpoint
    assert os.path.exists(ckpt_path), f"No checkpoint at {ckpt_path}"
    state = th.load(ckpt_path, map_location='cpu', weights_only=True)
    teacher.network.load_state_dict(state['model_state_dict'])
    teacher.to(dev)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"Loaded teacher from {ckpt_path}")

    # --- Distillation schedule ---
    start_N = config.pd.start_steps
    end_steps = config.pd.end_steps
    epochs_per_round = config.pd.epochs_per_round
    grad_accum = config.train.get("grad_accum_steps", 1)

    num_rounds = int(math.log2(start_N // end_steps))
    print(f"Progressive Distillation: {start_N} -> {end_steps} steps "
          f"({num_rounds} rounds, {epochs_per_round} epochs each)")

    # --- Weights & Biases ---
    wandb.init(
        project="darcy-pd",
        name=f"pd_exp{config.exp_num}",
        config=OmegaConf.to_container(config, resolve=True),
    )

    # --- Resume from a specific round ---
    start_round = 0
    if restart:
        start_round = config.train.checkpointing_dict.get("restart_round", 0)
        if start_round > 0:
            # Load the student from the previous round as teacher
            prev_steps = start_N // (2 ** start_round)
            prev_ckpt = f"{savepath}/pd_round{start_round - 1}_steps{prev_steps}.pt"
            assert os.path.exists(prev_ckpt), f"No checkpoint at {prev_ckpt}"
            prev_state = th.load(prev_ckpt, map_location='cpu', weights_only=True)
            teacher.network.load_state_dict(prev_state['model_state_dict'])
            teacher.to(dev)
            print(f"Resumed: loaded round {start_round - 1} student as teacher")

    # --- Progressive distillation rounds ---
    N = start_N // (2 ** start_round)  # Current teacher steps (accounting for resume)

    for round_idx in range(start_round, num_rounds):
        student_steps = N // 2
        print(f"\n{'='*60}")
        print(f"Round {round_idx}: {N} teacher steps -> {student_steps} student steps")
        print(f"{'='*60}")

        # Build student, initialize from teacher
        student_network = build_unet(config)
        student_network.load_state_dict(teacher.network.state_dict())
        student = VPDiffusionModel(
            network=student_network, schedule_s=schedule_s, ema_rate=0,
        )
        student.to(dev)

        # Loss and optimizer
        objective = ProgressiveDistillationLoss(teacher, N, schedule_s)
        optim = Adam(student.network.parameters(), lr=config.optimizer.lr)

        best_loss = float('inf')

        for epoch in tqdm(range(epochs_per_round),
                          desc=f"Round {round_idx} ({N}->{student_steps})"):
            student.network.train()
            total_loss = 0.0

            optim.zero_grad()
            for batch_idx, batch in enumerate(train_loader):
                loss = objective(student, batch, device=dev) / grad_accum
                loss.backward()

                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                    optim.step()
                    optim.zero_grad()

                total_loss += loss.item() * grad_accum

            avg_loss = total_loss / len(train_loader)
            best_loss = min(best_loss, avg_loss)

            wandb.log({
                "loss": avg_loss,
                "best_loss": best_loss,
                "round": round_idx,
                "teacher_steps": N,
                "student_steps": student_steps,
                "round_epoch": epoch,
                "global_epoch": round_idx * epochs_per_round + epoch,
            })

            if epoch % 10 == 0 or epoch == epochs_per_round - 1:
                tqdm.write(f"  Round {round_idx} Epoch {epoch}: "
                           f"loss={avg_loss:.6f}, best={best_loss:.6f}")

        # Save this round's student
        save_checkpoint(student, optim, round_idx, student_steps, savepath)
        print(f"  Saved pd_round{round_idx}_steps{student_steps}.pt")

        # Student becomes next teacher
        teacher = VPDiffusionModel(
            network=student_network, schedule_s=schedule_s, infer=True,
        )
        teacher.to(dev)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        N = student_steps  # Halve for next round

    print(f"\nProgressive Distillation complete!")
    print(f"Final student uses {N} DDIM steps")
    print(f"Checkpoints in {savepath}")
    wandb.finish()


if __name__ == '__main__':
    main(sys.argv[1])
