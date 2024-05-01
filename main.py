# %% Imports
import matplotlib.pyplot as plt
import torch_geometric
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import dense_to_sparse
from model import deep_GNN
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from torch_sparse import SparseTensor
import torch
import h5py
from torch import nn
import torch
import torch.nn.functional as F
import os
# %% - Define constants
OUTPUT_PATH = "output"
DATA_PATH = "./ccai_demo/data/FCN_ERA5_data_v0/out_of_sample"
DATA_FILE = os.path.join(DATA_PATH, "2018.h5")
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

N_LAYER = 3
N_HIDDEN = 64
DT = 1.
ALPHA = 1.
GAMMA = 1.
DROPOUT = 0.1
VARIABLES = ['u10',
             'v10',
             't2m',
             'sp',
             'msl',
             't850',
             'u1000',
             'v1000',
             'z1000',
             'u850',
             'v850',
             'z850',
             'u500',
             'v500',
             'z500',
             't500',
             'z50',
             'r500',
             'r850',
             'tcwv']
N_VAR = len(VARIABLES)

DEVICE = "mps" if torch.backends.mps.is_available(
) else "cuda" if torch.cuda.is_available() else "cpu"

# %% - Load data


class H5GeometricDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.file_path = path
        self.dataset = None
        with h5py.File(self.file_path, 'r') as file:
            # print keys
            print("Keys: %s" % file.keys())
            self.dataset_len = len(file["fields"])
            print(f"Dataset length: {self.dataset_len}")
        self.height = 721
        self.width = 1440
        self.features = 21
        self.edge_index = self.create_edge_index(self.height, self.width)
        self.adj = SparseTensor(row=self.edge_index[0], col=self.edge_index[1],
                                sparse_sizes=(self.height*self.width, self.height*self.width))

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')["fields"]
        data_array = self.dataset[index]
        data_array = data_array.reshape(-1, self.features)
        x = torch.tensor(data_array, dtype=torch.float)

        return x, self.adj

    def __len__(self):
        return self.dataset_len

    def create_edge_index(self, height, width):
        edge_index = grid(height=height, width=width)
        edge_index = torch_geometric.utils.to_undirected(edge_index)
        return edge_index


dataset = H5GeometricDataset(DATA_FILE)
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True)
# get sample to load the h5file
sample = dataset[0]
print(f"Sample shape: {sample.shape}")

# %% - plot sample
# unsqueeze batch dimension of numpy array
sample = torch.tensor(sample).unsqueeze(0)
timestep_idx = 0
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
for i, varname in enumerate(['u10', 't2m', 'z500', 'tcwv']):
    cm = 'bwr' if varname == 'u10' or varname == 'z500' else 'viridis'
    varidx = VARIABLES.index(varname)
    ax[i//2][i % 2].imshow(sample[timestep_idx, varidx], cmap=cm)
    ax[i//2][i % 2].set_title(varname)
fig.tight_layout()

# %% Evaluation
# define metrics from the definitions above


def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)


def latitude_weighting_factor(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s


def weighted_rmse_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each channel
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(
        lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1, -2)))
    return result


def weighted_acc_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc for each channel
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(
        lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1, -2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1, -2)) * torch.sum(weight * target *
                                                                                                                                    target, dim=(-1, -2)))
    return result

# %% - Define model


model = deep_GNN(nfeat=N_VAR, nhid=N_HIDDEN, nclass=N_VAR,
                 nlayers=N_LAYER, dt=DT, alpha=ALPHA, gamma=GAMMA, dropout=DROPOUT)

model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# %%
len(train_loader.dataset)
sample.shape
# %%
# data size is [batch, n_var, x, y] x = 721, y = 1440 for ERA5 n_var = 21
# we want a Graph with 721*1440 nodes and 21 features and each node is connected to its 8 neighbors


# create a grid graph
# TypeError: grid() missing 1 required positional argument: 'width'
(row, col), pos = torch_geometric.utils.grid(height=721, width=1440)
edge_index = torch.stack([row, col], dim=0)
edge_index = to_undirected(edge_index)
edge_index = edge_index.to(DEVICE)
sparse_adj = to_dense_adj(edge_index)
sparse_adj = dense_to_sparse(sparse_adj)
# edge indices stay always the same node feature is a time series of 21 features
# create a graph data object like H5Dataset for


# %%
