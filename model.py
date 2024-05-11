from utils.config import BATCH_SIZE, BATCH_SIZE_VAL, LEARNING_RATE, MODEL_CONFIG
from utils.eval import weighted_rmse_channels
from torch.optim import Adam
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn
from torch.utils.data import DistributedSampler
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader


class GraphCON(nn.Module):
    def __init__(self, GNNs, dt=1., alpha=1., gamma=1., dropout=None):
        super(GraphCON, self).__init__()
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma
        self.GNNs = GNNs  # list of the individual GNN layers
        self.dropout = dropout

    def forward(self, X0, Y0, edge_index):
        # set initial values of ODEs

        # solve ODEs using simple IMEX scheme
        for gnn in self.GNNs:
            Y0 = Y0 + self.dt * (torch.relu(gnn(X0, edge_index)) -
                                 self.alpha * Y0 - self.gamma * X0)
            X0 = X0 + self.dt * Y0

            if (self.dropout is not None):
                Y0 = F.dropout(Y0, self.dropout, training=self.training)
                X0 = F.dropout(X0, self.dropout, training=self.training)

        return X0, Y0


class deep_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, dt=1., alpha=1., gamma=1., dropout=None):
        super(deep_GNN, self).__init__()
        self.enc = nn.Linear(nfeat, nhid)
        self.GNNs = nn.ModuleList()
        for _ in range(nlayers):
            self.GNNs.append(GCNConv(nhid, nhid))
        self.graphcon = GraphCON(self.GNNs, dt, alpha, gamma, dropout)
        self.dec = nn.Linear(nhid, nclass)

    def forward(self, x0, edge_index):
        # compute initial values of ODEs (encode input)
        x0 = self.enc(x0)
        # stack GNNs using GraphCON
        x0, _ = self.graphcon(x0, x0, edge_index)
        # decode X state of GraphCON at final time for output nodes
        x0 = self.dec(x0)
        return x0


class LitModel(pl.LightningModule):
    def __init__(self, datasets, std, num_workers=1):
        super().__init__()
        self.model = deep_GNN(**MODEL_CONFIG)
        self.criteria = weighted_rmse_channels
        self.datasets = datasets
        self.num_cpus = num_workers
        self.std = std

    def forward(self, x, edge_index):
        # reshape?
        # edge_index are all the same
        edge_index = edge_index[0]
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx):
        x, edge_index, target = batch
        predictions = self(x, edge_index)
        loss = self.criteria(predictions, target)
        loss = (loss * self.std).mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def check_format(self, batch):
        # update std if needed to the device and dytpe of the batch
        if self.std.dtype != batch[0].dtype:
            print("Updating std to dtype")
            self.std = self.std.to(batch[0].dtype)
        if self.std.device != batch[0].device:
            print("Updating std to device")
            self.std = self.std.to(batch[0].device)

    def validation_step(self, batch, batch_idx):
        self.check_format(batch)
        x, edge_index, target = batch
        predictions = self(x, edge_index)
        loss = self.criteria(predictions, target)
        loss = (loss * self.std).mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def on_epoch_end(self):
        # Move to the next year after each epoch
        pass

    def train_dataloader(self):
        if self.on_gpu:
            sampler = DistributedSampler(self.datasets['train'], shuffle=True)
            loader = DataLoader(
                self.datasets['train'],
                batch_size=BATCH_SIZE,
                sampler=sampler,
                drop_last=True,
                num_workers=self.num_cpus,
                persistent_workers=True
            )
        else:
            loader = DataLoader(
                self.datasets['train'],
                batch_size=BATCH_SIZE,
                drop_last=True,
                shuffle=False,
                num_workers=self.num_cpus,
                persistent_workers=True
            )
        return loader

    def val_dataloader(self):
        # check if ddp is used
        if self.on_gpu:
            sampler = DistributedSampler(self.datasets['val'], shuffle=False)
            loader = DataLoader(
                self.datasets['val'],
                batch_size=BATCH_SIZE_VAL,
                sampler=sampler,
                drop_last=True,
                num_workers=self.num_cpus,
                persistent_workers=True
            )
        else:
            loader = DataLoader(
                self.datasets['val'],
                batch_size=BATCH_SIZE_VAL,
                drop_last=True,
                shuffle=False,
                num_workers=self.num_cpus,
                persistent_workers=True
            )
        return loader
