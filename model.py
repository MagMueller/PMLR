from utils.configs.config import BATCH_SIZE, BATCH_SIZE_VAL, LEARNING_RATE, MODEL_CONFIG
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


class LitModel(pl.LightningModule):
    def __init__(self, datasets, std, model, config, num_workers=1, stupid=False):
        super().__init__()
        self.batch_size = config['batch_size']
        self.batch_size_val = config['batch_size_val']
        self.n_var = config['n_var']
        self.height = config['height']
        self.width = config['width']
        self.lr = config['lr']

        self.model = model
        self.criteria = weighted_rmse_channels
        self.datasets = datasets
        self.num_cpus = num_workers
        self.std = std
        self.last_prediction = None
        self.count_autoreg_steps = 0
        self.stupid = stupid

    def forward(self, x, edge_index):
        # reshape?
        # edge_index are all the same
        edge_index = edge_index[0]
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx=0):
        x, edge_index, target = batch
        predictions = self(x, edge_index)
        loss = self.criteria(predictions, target, self.n_var, self.height, self.width)
        loss_scaled = (loss * self.std.squeeze()).mean()
        loss = (loss).mean()
        self.log(name='train_loss', value=loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(name='train_loss_scaled', value=loss_scaled, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        loss = self.criteria(predictions, target, self.n_var, self.height, self.width)
        loss_scaled = (loss * self.std.squeeze()).mean()
        loss = (loss).mean()
        self.log('val_loss_scaled', loss_scaled, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # autoregressive call model on itself and feed prediction back in, calculate loss with next timestep,
        # assume batch to be [batch_size, h*w, n_var]
        # assume batch size to be 1
        # save prediction with self.last_prediction and reuse it for the next timestep,
        # count
        self.check_format(batch)
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
