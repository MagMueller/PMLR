# %% Imports
import matplotlib.pyplot as plt
from scipy import sparse
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
BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 3
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
N_VAR = 21  # somehow the len here is just 20
print(f"Number of variables: {N_VAR}")
if torch.backends.mps.is_available():
    DEVICE = "mps"
    # set env var PYTORCH_ENABLE_MPS_FALLBACK=1
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# %% - Load data


class H5GeometricDataset(torch.utils.data.Dataset):
    # data size is [batch, n_var, x, y] x = 721, y = 1440 for ERA5 n_var = 21
    # we want a Graph with 721*1440 nodes and 21 features and each node is connected to its 8 neighbors
    def __init__(self, path, sequence_length=3):
        self.file_path = path
        self.dataset = None
        with h5py.File(self.file_path, 'r') as file:
            # print keys
            self.dataset_len = len(file["fields"])
        self.height = 721
        self.width = 1440
        self.features = 21
        # approx 800 Mio values -> 3.2 GB
        self.sequence_length = sequence_length
        self.edge_index = self.create_edge_index(self.height, self.width)
        # TODO to sparse

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')["fields"]
        # load a sequence of 5 time steps
        sequences = [self.dataset[i].reshape(-1, self.features)
                     for i in range(index, index + self.sequence_length + 1)]
        # numpy array
        sequences = np.stack(sequences, axis=0)
        # to torch tensor
        x = torch.tensor(sequences, dtype=torch.float32).to(DEVICE)
        return x, self.edge_index

    def __len__(self):
        return self.dataset_len - self.sequence_length

    def create_edge_index(self, height, width):
        # 721*1440*9 = 9344160 -> 9331198 edges (not at the border)
        (row, col), pos = grid(height=height, width=width)
        edge_index = torch.stack([row, col], dim=0).to(torch.long).to(DEVICE)
        edge_index = to_undirected(edge_index)
        # edge_index = edge_index.to_sparse()
        # edge_index = edge_index.coalesce()
        return edge_index


dataset = H5GeometricDataset(DATA_FILE, sequence_length=SEQUENCE_LENGTH)
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True)
# %% # get sample
x, adj = next(iter(train_loader))
print(f"x shape: {x.shape}, adj shape: {adj.shape}")
a = x.element_size() * x.nelement() / 1024**2
print(F"Memory size of x: {a} MB")
a = adj.element_size() * adj.nelement() / 1024**2
print(F"Memory size of adj: {a} MB")

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


# %% - Train model
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch, data in enumerate(loader):
        x, edge_index = data
        # Shape: [batch_size, sequence_length, num_nodes, num_features]
        x = x.to(device)
        edge_index = edge_index.to(device).squeeze()
        print(f"X shape: {x.shape}, edge_index shape: {edge_index.shape}")
        # Reshape to fit the model's expected input and prepare for rolling prediction
        batch_size, seq_len, num_nodes, num_features = x.shape
        losses = []

        optimizer.zero_grad()

        # Iterate over each timestep, predicting the next state
        for t in range(seq_len - 1):
            x_input = x[:, t, :, :].view(batch_size, num_nodes, num_features)
            x_target = x[:, t + 1, :,
                         :].view(batch_size, num_nodes, num_features)

            # Model output for current timestep
            predictions = model(x_input, edge_index)

            # Compute loss for the current timestep prediction
            loss = criterion(predictions, x_target)
            losses.append(loss)
            print(f"Loss: {loss.item()} at timestep {t}")
        # Average loss across the sequence for backpropagation
        sequence_loss = sum(losses) / len(losses)
        sequence_loss.backward()
        optimizer.step()
        total_loss += sequence_loss.item()
        print(f"Total loss: {total_loss} at batch {batch}")

    return total_loss / len(loader)


criteria = nn.MSELoss()
total_loss = train(model, train_loader, optimizer, criteria, DEVICE)


# %%

# eval
