# %%
from utils.configs.config import N_VAR
from utils.configs.config import VARIABLES
import numpy as np
from utils.configs.config import OUTPUT_FILE
import json
import time
import matplotlib.pyplot as plt
from model import deep_GNN
import torch
from torch import nn
import torch
import os
from utils.dataset import H5GraphDataset
from utils.eval import evaluate
from utils.train import train_one_epoch
from utils.configs.config import *
import wandb


# %% - Load data
# Device Configuration
if torch.backends.mps.is_available():
    DEVICE = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# create validation dataset
val_dataset = H5GraphDataset(VAL_FILE, sequence_length=SEQUENCE_LENGTH_VAL, height=HEIGHT, width=WIDTH, features=N_VAR)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
wandb.init(project="PMLR")

# %% load model
model = deep_GNN(**MODEL_CONFIG)
model.to(DEVICE)
# wandb checkpoint
run = wandb.init()
file = "v42"
file = "latest"
artifact = run.use_artifact('forl-traffic/PMLR/model-4yobs5ix:' + file, type='model')
artifact_dir = artifact.download()

# %%
model.load_state_dict(torch.load(os.path.join(artifact_dir, 'best_model.pt'), map_location=DEVICE))
# %%


# %%
# evaluate
print(f"Model evaluation")
evaluate(model, validation_loader, DEVICE, subset=SUBSET_VAL, autoreg=True, log=False, prediction_len=PREDICTION_LENGTH, name="weather_graph_corrn" + file)


# %%
print(f"Stupid model evaluation: predict always input")
evaluate(None, validation_loader, DEVICE, subset=SUBSET_VAL, autoreg=True, log=False, prediction_len=PREDICTION_LENGTH, name="stupid")


# %%
# load output json
with open(OUTPUT_FILE) as f:
    data = json.load(f)

# print len of each
for key in data.keys():
    print(f"Key: {key} - len: {len(data[key]['acc'])}")

# %%
# plot mean -1

# loop over data and plot in one plot all model over all variables mean -1
# make a beautiful plot
plt.figure(figsize=(20, 10))
plt.title("Mean accuracy over all variables")
plt.xlabel("Days")
step = 6  # 1 value equals 6 hours

# devide x ticks by 4 to get days
plt.xticks(np.arange(0, PREDICTION_LENGTH, step), np.arange(0, PREDICTION_LENGTH, step)/4)
for key in data.keys():
    plt.plot(np.array(data[key]['acc']).mean(axis=-1), label=key)
plt.legend()

# %%
# rmse
plt.figure(figsize=(20, 10))
plt.title("Mean RMSE over all variables")
plt.xlabel("Days")
step = 6  # 1 value equals 6 hours

# devide x ticks by 4 to get days
plt.xticks(np.arange(0, PREDICTION_LENGTH, step), np.arange(0, PREDICTION_LENGTH, step)/4)
for key in data.keys():
    plt.plot(np.array(data[key]['rmse']).mean(axis=-1), label=key)
plt.legend()

# %%
# loop over every variable and plot the mean accuracy in one plot
var = "rmse"  # or "acc", "rmse"
plt.figure(figsize=(20, 10))
plt.title("Mean accuracy over all variables")
plt.xlabel("Days")
# all models i one plot
for i in range(N_VAR):
    plt.plot(np.array(data['weather_graph_corrn'][var])[:, i], label=VARIABLES[i])
plt.legend()

# %%

var = "acc"  # or "acc", "rmse"
models = list(data.keys())
plt.figure(figsize=(20, 10))
# bar plot x axis is the variable and y axis is the mean over all prediction steps
# all models i one plot
for model in models:
    means = np.array(data[model][var]).mean(axis=0)
    plt.bar(VARIABLES, means, alpha=0.5, label=model)
plt.title(f"Mean {var} over all variables")
plt.xlabel("Variable")
plt.ylabel("Mean")
plt.xticks(rotation=90)
plt.legend()
plt.show()

# %%
# now similar to bar plot, but models next to each other with some distance
var = "rmse"  # or "acc", "rmse"
models = list(data.keys())
plt.figure(figsize=(20, 10))
for i, model in enumerate(models):
    means = np.array(data[model][var]).mean(axis=0)
    plt.bar(np.arange(N_VAR) + i * 0.3, means, alpha=0.5, label=model)
plt.title(f"Mean {var} over all variables")
plt.xlabel("Variable")
plt.ylabel("Mean")
plt.xticks(np.arange(N_VAR), VARIABLES, rotation=90)
plt.legend()
plt.show()
# %%
# now each model in individual plot
var = "rmse"  # or "acc", "rmse"
models = list(data.keys())
plt.figure(figsize=(20, 10))
for model in models:
    plt.figure(figsize=(20, 10))
    means = np.array(data[model][var]).mean(axis=0)
    plt.bar(VARIABLES, means, alpha=0.5, label=model)
    plt.title(f"Mean {var} over all variables")
    plt.xlabel("Variable")
    plt.ylabel("Mean")
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
# %%
