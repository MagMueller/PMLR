
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
from utils.config import *
import wandb


wandb.init(project="PMLR")

# %% - Load data

datasets = []
train_loaders = []
for year in YEARS:
    data_file = os.path.join(DATA_FILE_PATH, f"{year}.h5")
    dataset = H5GeometricDataset(data_file, sequence_length=SEQUENCE_LENGTH, height=HEIGHT, width=WIDTH, features=N_VAR)
    datasets.append(dataset)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loaders.append(train_loader)

# create validation dataset
val_dataset = H5GeometricDataset(VAL_FILE, sequence_length=SEQUENCE_LENGTH_VAL, height=HEIGHT, width=WIDTH, features=N_VAR)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False)

# %%
# len(train_loader)
# for batch in train_loader:
#     # shape
#     x, edge_index = batch
#     print(f"x shape: {x.shape}, edge_index shape: {edge_index.shape}")
#     break
# # %% -
# # print sample
# data_sample = datasets[0][0]
# print(len(data_sample))
# print(data_sample[0][:][0])
# print(data_sample[0].shape)
# print(data_sample[1].shape)
# # %% - further testing
# data_testing = datasets[0]
# print(len(data_testing))
# print(len(data_testing[0]))
# print(len(data_testing[0][0]))
# print(len(data_testing[0][0][0]))
# print(len(data_testing[0][0][0][0]))
# print(data_testing[0][0][0][100001])
# print(data_testing[0][0][0][100000])

# %% - calculate size of dataset class in mb

model = deep_GNN(nfeat=N_VAR, nhid=N_HIDDEN, nclass=N_VAR, nlayers=N_LAYER, dt=DT, alpha=ALPHA, gamma=GAMMA, dropout=DROPOUT)

# Use DataParallel to wrap the model
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(DEVICE)
wandb.watch(model, log="all", log_freq=100)

# # get model size
# model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
# model_size = model_size * 4 / 1024**2
# print(f"Model size: {model_size:0.2f} MB")

# # calc number of parameters
# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Number of parameters: {num_params}")
# # check - TODO: whats the right calculation?
# manuel_size = N_HIDDEN * N_VAR * 9
# print(f"Size of model: {manuel_size}")

# size in mb of val sample, 0 is x and 1 is edge_index
start = time.time()

# val_sample = next(iter(validation_loader))
# x, edge_index = val_sample
# val_size = x.numel() * x.element_size() / 1024**2
# print(f"Size of validation sample: {val_size:0.2f} MB")
# edge_size = edge_index.numel() * edge_index.element_size() / 1024**2
# print(f"Size of edge_index: {edge_size:0.2f} MB")
# print(f"time to load train: {(time.time() - start) / 60:.02f} min")

# training sample in mb
start = time.time()
# train_sample = next(iter(train_loaders[0]))
# x, edge_index = train_sample
# train_size = x.numel() * x.element_size() / 1024**2
# print(f"\nSize of training sample: {train_size:0.2f} MB")
# edge_size = edge_index.numel() * edge_index.element_size() / 1024**2
# print(f"Size of edge_index: {edge_size:0.2f} MB")
# print(f"time to load train: {(time.time() - start) / 60:.02f} min")
# %% - Train model

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criteria = nn.MSELoss()
best_acc = 0

# %%
# inital evaluation
print(f"\nEvaluating initial model...")
rmse, acc = evaluate(model, validation_loader, DEVICE, subset=SUBSET_VAL, prediction_len=PREDICTION_LENGTH)


# %%
# train model for every year with seperate dataloader
# torch cleanup mps
# torch.cuda.empty_cache()
print(f"\n\n\nTraining model...")
for epoch in range(EPOCHS):
    print(f"\n\n\nEpoch {epoch + 1}/{EPOCHS}")

    for i in range(len(YEARS)):
        print(f"\nTraining ...")

        train_loader = train_loaders[i]

        train_loss = train_one_epoch(model, train_loader, optimizer, criteria, DEVICE, subset=SUBSET_TRAIN)
        print(f"Train loss: {train_loss}")

        # eval
        # Currently no test set: DONE
        print(f"\nEvaluating...")
        # print(f"val loader: {validation_loader}")
        rmse, acc = evaluate(model, validation_loader, DEVICE, subset=SUBSET_VAL, prediction_len=PREDICTION_LENGTH)
        wandb.log({"epoch": epoch + 1})

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, "best_model.pt"))
            # artifact
            artifact = wandb.Artifact('best_model', type='model')
            artifact.add_file(os.path.join(OUTPUT_PATH, "best_model.pt"))
            wandb.log_artifact(artifact)


# %%
