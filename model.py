import einops
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
from deep_GNN import deep_GNN
import numpy as np


class LitModel(pl.LightningModule):
    def __init__(self, datasets, model, cfg):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.batch_size_val = cfg.batch_size_val
        self.n_var = cfg.n_var
        self.height = cfg.height
        self.width = cfg.width
        self.lr = cfg.lr

        self.model = model
        self.criteria = weighted_rmse_channels
        self.datasets = datasets
        self.num_cpus = cfg.env.num_workers
        self.stupid = cfg.stupid
        self.std = torch.tensor(np.load(cfg.global_stds)[0, :cfg.n_var]).unsqueeze(0)
        self.last_prediction = None
        self.count_autoreg_steps = 0
        self.model_name = self.model.__class__.__name__

    def forward(self, x, edge_index=None):
        if edge_index is None:
            return self.model(x)
        else:
            return self.model(x, edge_index)

    def convert_std(self, batch):
        # update std if needed to the device and dytpe of the batch
        if self.std.dtype != batch[0].dtype:
            print("Updating std to dtype")
            self.std = self.std.to(batch[0].dtype)
        if self.std.device != batch[0].device:
            print("Updating std to device")
            self.std = self.std.to(batch[0].device)

    def training_step(self, batch, batch_idx=0):

        if self.model_name == "deep_GNN":
            loss, loss_scaled = self.step_gnn(batch)
        elif self.model_name == "coRNN" or self.model_name == "coRNN2":
            loss, loss_scaled = self.step_image(batch)
        else:
            raise ValueError(f"Model {self.model_name} not implemented")

        self.log(name='train_loss', value=loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(name='train_loss_scaled', value=loss_scaled, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def step_gnn(self, batch):
        x, edge_index, target = batch
        edge_index = edge_index[0]
        predictions = self(x, edge_index)
        loss = self.criteria(predictions, target, self.n_var, self.height, self.width)
        loss_scaled = (loss * self.std.squeeze()).mean()
        loss = (loss).mean()
        return loss, loss_scaled

    def step_image(self, batch):
        # batch has shape (B, Seq_len, n_var, H, W)
        # loop over sequence and predict always next, calculate loss over all

        total_loss = 0
        total_loss_scaled = 0
        # last target is the target for the next step
        batch = einops.rearrange(batch, 'b t c h w -> t b c h w')

        target = batch[-1]
        x = batch[:-1]
        # swithch batch and seq_len
        predictions = self(x)
        loss = self.criteria(predictions, target, self.n_var, self.height, self.width)
        loss_scaled = (loss * self.std.squeeze()).mean()
        loss = (loss).mean()
        total_loss += loss
        total_loss_scaled += loss_scaled

        return total_loss, total_loss_scaled

    def validation_step(self, batch, batch_idx):
        self.convert_std(batch)

        if self.model_name == "deep_GNN":
            loss, loss_scaled = self.step_gnn(batch)
        elif self.model_name == "coRNN" or self.model_name == "coRNN2":
            loss, loss_scaled = self.step_image(batch)
        else:
            raise ValueError(f"Model {self.model_name} not implemented")

        self.log('val_loss_scaled', loss_scaled, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # autoregressive call model on itself and feed prediction back in, calculate loss with next timestep,
        # assume batch to be [batch_size, h*w, n_var]
        # assume batch size to be 1
        # save prediction with self.last_prediction and reuse it for the next timestep,
        # count
        self.convert_std(batch)
        x, edge_index, target = batch

        if self.count_autoreg_steps != 0:
            x = self.last_prediction

        if self.stupid:
            predictions = x
        else:
            predictions = self(x, edge_index)

        loss = self.criteria(predictions, target, self.n_var, self.height, self.width)
        loss_scaled = (loss * self.std.squeeze()).mean()
        loss = (loss).mean()
        self.log('autoreg_rmse_scaled', loss_scaled, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.last_prediction = predictions
        self.log('autoreg_rmse', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.count_autoreg_steps += 1
        print(f"Test step: {self.count_autoreg_steps}")
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_epoch_end(self):
        # Move to the next year after each epoch
        pass

    def train_dataloader(self):
        loader = DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=self.num_cpus,
            persistent_workers=True
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.datasets['val'],
            batch_size=self.batch_size_val,
            drop_last=True,
            shuffle=False,
            num_workers=self.num_cpus,
            persistent_workers=True
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.datasets['val'],
            batch_size=self.batch_size_val,
            drop_last=True,
            shuffle=False,
            num_workers=0,
        )
        return loader
