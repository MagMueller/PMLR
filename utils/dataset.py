
import h5py
import os
import numpy as np
import torch
from torch_geometric.utils import to_undirected
from torch_geometric.utils import grid
from utils.config import DEVICE


class H5GeometricDataset(torch.utils.data.Dataset):
    # data size is [batch, n_var, x, y] x = 721, y = 1440 for ERA5 n_var = 21
    # we want a Graph with 721*1440 nodes and 21 features and each node is connected to its 8 neighbors
    def __init__(self, path, sequence_length=3, height=721, width=1440, features=21):
        self.file_path = path
        self.dataset = None

        # check if file exists
        if not os.path.exists(self.file_path):
            # download file
            # ```bash
            # wget https://portal.nersc.gov/project/m4134/ccai_demo.tar
            # tar -xvf ccai_demo.tar
            # rm ccai_demo.tar
            # ```
            print(f"File {self.file_path} does not exist we will download it now")
            import urllib.request
            url = "https://portal.nersc.gov/project/m4134/ccai_demo.tar"
            urllib.request.urlretrieve(url, "ccai_demo.tar")
            os.system("tar -xvf ccai_demo.tar")
            os.system("rm ccai_demo.tar")
            print("Downloaded and extracted file successfully")
            # approx 800 Mio values -> 3.2 GB

        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["fields"])

        self.height = height
        self.width = width
        self.features = features
        self.sequence_length = sequence_length
        self.edge_index = self.create_edge_index(self.height, self.width)
        # TODO to sparse

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')["fields"]
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
        # edge_index = edge_index.to_sparse() # not implemented on mps for mac
        # edge_index = edge_index.coalesce()
        return edge_index
