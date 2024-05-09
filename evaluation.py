# %%
import time
import matplotlib.pyplot as plt
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
wandb.init(project="PMLR")


# %% - Load data

# create validation dataset
val_dataset = H5GeometricDataset(VAL_FILE, sequence_length=SEQUENCE_LENGTH_VAL, height=HEIGHT, width=WIDTH, features=N_VAR)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# %% load model
model = deep_GNN(nfeat=N_VAR, nhid=N_HIDDEN, nclass=N_VAR, nlayers=N_LAYER, dt=DT, alpha=ALPHA, gamma=GAMMA, dropout=DROPOUT)
model.to(DEVICE)
# wandb checkpoint
run = wandb.init()
artifact = run.use_artifact('forl-traffic/PMLR/best_model:v39', type='model')
artifact_dir = artifact.download()

# %%
model.load_state_dict(torch.load(os.path.join(artifact_dir, 'best_model.pt'), map_location=DEVICE))
# %%
print(len(validation_loader))


# %%
# evaluate
evaluate(model, validation_loader, DEVICE, subset=SUBSET_VAL, autoreg=True, log=True, prediction_len=PREDICTION_LENGTH)

# %%
print(f"Stupid model evaluation: predict always input")
evaluate(None, validation_loader, DEVICE, subset=SUBSET_VAL, autoreg=True, log=False, prediction_len=PREDICTION_LENGTH)

# %%
