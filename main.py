
# %% Imports
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
# %% - Define constants
from utils.config import *


# %% - Load data


dataset = H5GeometricDataset(
    DATA_FILE, sequence_length=SEQUENCE_LENGTH, height=HEIGHT, width=WIDTH, features=N_VAR)
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True)
# %% # get sample
x, adj = next(iter(train_loader))
print(f"x shape: {x.shape}, adj shape: {adj.shape}")
print(
    F"Memory size of x: {(x.element_size() * x.nelement() / 1024**2):0.2f} MB for sequence length {SEQUENCE_LENGTH}")
print(
    F"Memory size of adj: {(adj.element_size() * adj.nelement() / 1024**2):0.2f} MB")


# %% - Define model


model = deep_GNN(nfeat=N_VAR, nhid=N_HIDDEN, nclass=N_VAR,
                 nlayers=N_LAYER, dt=DT, alpha=ALPHA, gamma=GAMMA, dropout=DROPOUT)
model.to(DEVICE)
# get model size
model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size = model_size * 4 / 1024**2
print(f"Model size: {model_size:0.2f} MB")

# %% - Train model


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criteria = nn.MSELoss()
best_acc = 0
for epoch in range(EPOCHS):
    print(f"\n\n\nEpoch {epoch + 1}/{EPOCHS}")

    print(f"\nTraining ...")
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criteria, DEVICE, subset=2)
    print(f"Train loss: {train_loss}")

    # eval
    # Currently no test set
    print(f"\nEvaluating ...")
    rmse, acc = evaluate(model, train_loader, DEVICE, subset=2)
    # print(f"RMSE: {rmse}, ACC: {acc}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(
            OUTPUT_PATH, "best_model.pt"))
    # save model

# %%


# %%
