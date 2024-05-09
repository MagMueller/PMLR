import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import torch
from torch import nn


def train_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str, subset=None) -> float:
    start_time = time.time()
    model.train()
    total_loss = 0
    total_batches = len(loader) if subset is None else subset
    for batch, (x, edge_index) in enumerate(loader):
        if subset is not None and batch >= subset:
            break
        batch_start_time = time.time()

        # Shape: [batch_size, sequence_length, num_nodes, num_features]
        x = x.to(device)
        # assume all batches have the same edge_index -> take the first one
        edge_index = edge_index.to(device)[0].squeeze()

        # Reshape to fit the model's expected input and prepare for rolling prediction
        batch_size, seq_len, num_nodes, num_features = x.shape
        losses = []

        optimizer.zero_grad()

        # Iterate over each timestep, predicting the next state
        for t in range(seq_len - 1):
            x_input = x[:, t, :, :].view(batch_size, num_nodes, num_features)
            x_target = x[:, t + 1, :, :].view(batch_size, num_nodes, num_features)
            del x
            # Model output for current timestep
            predictions = model(x_input, edge_index)
            del x_input
            # Compute loss for the current timestep prediction
            loss = criterion(predictions, x_target)
            losses.append(loss)

        # Average loss across the sequence for backpropagation
        sequence_loss = sum(losses) / len(losses)
        sequence_loss.backward()
        optimizer.step()
        total_loss += sequence_loss.item()
        if batch % 100 == 0:
            print(f"Sample {batch + 1}/{total_batches}  - Loss: {sequence_loss.item()} - Time taken: {(time.time() - batch_start_time):.2f} seconds")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")

    return total_loss / len(loader)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()
