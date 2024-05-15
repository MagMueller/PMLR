
# %% Imports

from ast import arg
from pytorch_lightning.loggers import WandbLogger

import time

from model import deep_GNN
import torch
from torch import device, nn
import torch
import os
from utils.dataset import H5GeometricDataset
from utils.eval import evaluate
from utils.train import train_one_epoch
from utils.configs.config import *
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from model import LitModel
import numpy as np
from torch.utils.data import ConcatDataset
from pytorch_lightning.callbacks import ModelCheckpoint


def main(args):

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,
        mode='min'
    )
    run_name = ""
    if args.eval:
        run_name = "eval_model"
        if args.stupid:
            run_name = "eval_stupid"
        wandb_logger = WandbLogger(project='PMLR', name=run_name, log_model="all")
    else:
        wandb_logger = WandbLogger(project='PMLR', log_model="all")

    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=EPOCHS,
        strategy=STRATEGY,
        num_nodes=args.nodes,
        devices=args.devices,
        precision=32,
        # limit_train_batches=12,
        # limit_val_batches=12,
        accumulate_grad_batches=4,
        profiler="simple",
        callbacks=[checkpoint_callback]
    )
    print(f"Number of devices: {trainer.num_devices}")

    # for normalizing
    TIME_MEANS = np.load(TIME_MEANS_PATH)[0, :N_VAR]
    MEANS = np.load(GLOBAL_MEANS_PATH)[0, :N_VAR]
    STDS = np.load(GLOBAL_STDS_PATH)[0, :N_VAR]
    M = torch.as_tensor((TIME_MEANS - MEANS)/STDS)[:, 0:HEIGHT].unsqueeze(0)
    STD = torch.tensor(STDS).unsqueeze(0)
    datasets = {
        "train": ConcatDataset([H5GeometricDataset(os.path.join(DATA_FILE_PATH, f"{year}.h5"), means=MEANS, stds=STDS) for year in YEARS]),
        "val": H5GeometricDataset(VAL_FILE, means=MEANS, stds=STDS)
    }

    # wandb_logger.watch(model, log='all', log_freq=100)

    if args.eval:
        # use script download_artifact.py to download the model
        path = "artifacts/model-4yobs5ix:v9"
        stupid = args.stupid

        model = LitModel.load_from_checkpoint(os.path.join(path, "model.ckpt"), datasets=datasets, num_workers=args.devices, std=STD, map_location=DEVICE, stupid=stupid)
        print(f"autoregressive step is {model.count_autoreg_steps}")
        print(f"Model evaluation")
        trainer.test(model)
        return
    else:
        model = LitModel(datasets=datasets, num_workers=args.devices, std=STD)

        wandb_logger.watch(model, log='all', log_freq=100)
        trainer.fit(model)


# %%
if __name__ == "__main__":

    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--devices", type=int, default=1)
    argparser.add_argument("--nodes", type=int, default=1)
    argparser.add_argument("--eval", action="store_true")
    argparser.add_argument("--stupid", action="store_true")
    args = argparser.parse_args()

    main(args)
