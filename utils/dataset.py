
from torch.utils.data.distributed import DistributedSampler
import h5py
import os
import numpy as np
import torch
from torch_geometric.utils import to_undirected
from torch_geometric.utils import grid
from torch.utils.data import Dataset, ConcatDataset


class H5GraphDataset(torch.utils.data.Dataset):
    # data size is [batch, n_var, x, y] x = 721, y = 1440 for ERA5 n_var = 21
    # we want a Graph with 721*1440 nodes and 21 features and each node is connected to its 8 neighbors
    def __init__(self, path, cfg):
        self.file_path = path
        self.dataset = None

        # check if file exists
        if not os.path.exists(self.file_path):
            print(f"File {self.file_path} does not exist you can download it with the following commands: \n wget https://portal.nersc.gov/project/m4134/ccai_demo.tar \n tar -xvf ccai_demo.tar \n rm ccai_demo.tar\n")
            # # approx 800 Mio values -> 3.2 GB

        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["fields"])

        self.height = cfg.height
        self.width = cfg.width
        self.features = cfg.n_var

        self.edge_index = self.create_edge_index(self.height, self.width)

        # load means and stds
        self.means = np.load(cfg.env.global_means)[0, :cfg.n_var].squeeze()
        self.stds = np.load(cfg.env.global_stds)[0, :cfg.n_var].squeeze()
        # TODO to sparse

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')["fields"]
        x = self.dataset[index, :, :, :].transpose(1, 2, 0)[:, :, :-1].reshape(-1, self.features)
        target = self.dataset[index + 1, :, :, :].transpose(1, 2, 0)[:, :, :-1].reshape(-1, self.features)

        x = (x - self.means) / self.stds
        target = (target - self.means) / self.stds

        # to torch tensor
        x = torch.tensor(x, dtype=torch.float32)  # .to(DEVICE)
        target = torch.tensor(target, dtype=torch.float32)

        # remove sequence length dimension
        x = x.squeeze()
        target = target.squeeze()
        return x, self.edge_index, target

    def __len__(self):
        return self.dataset_len - 1

    def create_edge_index(self, height, width):

        # 721*1440*9 = 9344160 -> 9331198 edges (not at the border)
        (row, col), pos = grid(height=height, width=width)
        edge_index = torch.stack([row, col], dim=0)  # .to(torch.long).to(DEVICE)
        edge_index = to_undirected(edge_index)
        # edge_index = edge_index.to_sparse() # not implemented on mps for mac
        # edge_index = edge_index.coalesce()
        return edge_index


class H5ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, cfg):
        self.file_path = path
        self.dataset = None

        # check if file exists
        if not os.path.exists(self.file_path):
            print(f"File {self.file_path} does not exist you can download it with the following commands: \n wget https://portal.nersc.gov/project/m4134/ccai_demo.tar \n tar -xvf ccai_demo.tar \n rm ccai_demo.tar\n")
            # # approx 800 Mio values -> 3.2 GB
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["fields"])

        self.height = cfg.height
        self.width = cfg.width
        self.features = cfg.n_var

        # load means and stds
        self.means = np.load(cfg.env.global_means)[0, :cfg.n_var].squeeze().reshape(1, -1, 1, 1)
        self.stds = np.load(cfg.env.global_stds)[0, :cfg.n_var].squeeze().reshape(1, -1, 1, 1)
        self.seq_len = cfg.sequence_length

    def __len__(self):
        return self.dataset_len - self.seq_len - 1

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')["fields"]

        # read sequence of data and return the normalized sequence with seq_len + 1 for target

        out = self.dataset[index:index + self.seq_len + 1, :-1]

        out = (out - self.means) / self.stds

        return torch.tensor(out, dtype=torch.float32)
