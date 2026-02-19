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

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.utils.dataloader import get_loaders_vf_fm
from src.utils.dataset import DATASETS
from src.training.trainer import Trainer
from src.training.objectives import VPDiffusionLoss
from src.models.vp_diffusion import VPDiffusionModel


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


def main(config_path):
    config = OmegaConf.load(config_path)

    th.manual_seed(config.th_seed)
    np.random.seed(config.np_seed)

    logpath = config.path + f"/exp_{config.exp_num}"
    savepath = logpath + "/saved_state"
    create_dir(logpath, config=config)
    create_dir(savepath, config=config)

    config.train.logger_dict.dir = logpath
    config.train.checkpointing_dict.save_path = savepath

    dev = th.device(config.device)

    train_dataloader = get_loaders_vf_fm(
        vf_paths=config.dataloader.datapath,
        batch_size=config.dataloader.batch_size,
        dataset_=DATASETS[config.dataloader.dataset],
    )

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
        class_cond=config.unet.class_cond if hasattr(config.unet, 'class_cond') else False,
        num_classes=config.unet.num_classes if hasattr(config.unet, 'num_classes') else None,
    )

    schedule_s = config.vp_diffusion.schedule_s if hasattr(config, 'vp_diffusion') else 0.008

    model = VPDiffusionModel(
        network=network,
        schedule_s=schedule_s,
    )

    optim = Adam(model.network.parameters(), lr=config.optimizer.lr)
    sched = None

    class_cond = config.unet.class_cond if hasattr(config.unet, 'class_cond') else False
    objective = VPDiffusionLoss(class_conditional=class_cond)

    trainer = Trainer(
        model=model,
        objective=objective,
        dataloader=train_dataloader,
        optimizer=optim,
        scheduler=sched,
        logger_dict=config.train.logger_dict,
        checkpointing_dict=config.train.checkpointing_dict,
        device=dev,
        config=config,
    )

    trainer.train(num_epochs=config.train.num_epochs)


if __name__ == '__main__':
    main(sys.argv[1])
