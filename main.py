
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
from utils.config import *
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


    wandb_logger = WandbLogger(project='PMLR', log_model="all")


    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=EPOCHS,
        strategy=STRATEGY,
        num_nodes= args.nodes,
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

    model = LitModel(datasets=datasets, num_workers= args.devices, std=STD)
    # wandb_logger.watch(model, log='all', log_freq=100)
    wandb_logger.watch(model, log='all', log_freq=100)
    trainer.fit(model)


# %%
if __name__ == "__main__":

    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--devices", type=int, default=1)
    argparser.add_argument("--nodes", type=int, default=1)
    args = argparser.parse_args()

    main(args)
