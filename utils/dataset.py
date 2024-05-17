
from torch.utils.data.distributed import DistributedSampler
import h5py
import os
import numpy as np
import torch
from torch_geometric.utils import to_undirected
from torch_geometric.utils import grid
from torch.utils.data import Dataset, ConcatDataset

from utils.configs.config import HEIGHT, N_VAR, WIDTH


class H5GeometricDataset(torch.utils.data.Dataset):
    # data size is [batch, n_var, x, y] x = 721, y = 1440 for ERA5 n_var = 21
    # we want a Graph with 721*1440 nodes and 21 features and each node is connected to its 8 neighbors
    def __init__(self, path,  means=None, stds=None, Y1=0, Y2=721, X1=0, X2=1440):
        self.file_path = path
        self.dataset = None
        self.Y1 = Y1
        self.Y2 = Y2
        self.X1 = X1
        self.X2 = X2

        # check if file exists
        if not os.path.exists(self.file_path):
            print(f"File {self.file_path} does not exist we will download it now")
            # download file
            # ```bash
            # wget https://portal.nersc.gov/project/m4134/ccai_demo.tar
            # tar -xvf ccai_demo.tar
            # rm ccai_demo.tar
            # ```
            # # approx 800 Mio values -> 3.2 GB

        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["fields"])

        self.height = self.Y2 - self.Y1
        self.width = self.X2 - self.X1
        self.features = N_VAR

        self.edge_index = self.create_edge_index(self.height, self.width)

        if means is None or stds is None:
            self.means = None
            self.stds = None
        else:
            self.means = means.squeeze()
            self.stds = stds.squeeze()
        # TODO to sparse

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')["fields"]
        x = self.dataset[index, :, self.Y1:self.Y2, self.X1:self.X2].transpose(1, 2, 0)[:, :, :-1].reshape(-1, self.features)
        # np
        target = self.dataset[index + 1, :, self.Y1:self.Y2, self.X1:self.X2].transpose(1, 2, 0)[:, :, :-1].reshape(-1, self.features)

        # normalizing
        if self.means is not None and self.stds is not None:
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


class CustomConcatDataset(ConcatDataset):
    def set_indices(self, Y1, Y2, X1, X2):
        for dataset in self.datasets:
            dataset.set_indices(Y1, Y2, X1, X2)
