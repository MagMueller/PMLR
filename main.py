
# %% Imports

from pytorch_lightning.loggers import WandbLogger

import time
from model import deep_GNN
import torch
from torch import nn
import torch
import os
from utils.dataset import H5GeometricDataset
from utils.eval import evaluate
from utils.train import train_one_epoch
from utils.config import *
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from model import LitModel
import numpy as np
from torch.utils.data import ConcatDataset


def main():
    # wandb.init(project="PMLR")

    wandb_logger = WandbLogger(name='test', project='PMLR')
    if LOCAL:
        strategy = "auto"
    else:
        strategy = 'ddp'

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=10,
        strategy=strategy
    )

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

    model = LitModel(datasets=datasets, num_workers=2, std=STD)
    # wandb_logger.watch(model, log='all', log_freq=100)
    trainer.fit(model)


# %%
if __name__ == "__main__":
    main()
