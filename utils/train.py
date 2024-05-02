import time
import torch
from torch import nn


def train_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str, subset=None) -> float:
    start_time = time.time()
    model.train()
    total_loss = 0
    total_batches = len(loader) if subset is None else subset
    for batch, data in enumerate(loader):
        if subset is not None and batch >= subset:
            break
        batch_start_time = time.time()
        x, edge_index = data
        # Shape: [batch_size, sequence_length, num_nodes, num_features]
        x = x.to(device)
        edge_index = edge_index.to(device).squeeze()
        # print(f"X shape: {x.shape}, edge_index shape: {edge_index.shape}")
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

        # Average loss across the sequence for backpropagation
        sequence_loss = sum(losses) / len(losses)
        sequence_loss.backward()
        optimizer.step()
        total_loss += sequence_loss.item()
        print(
            f"Sample {batch + 1}/{total_batches}  - Loss: {sequence_loss.item()} - Time taken: {(time.time() - batch_start_time):.2f} seconds - Sequence length {seq_len}")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")

    return total_loss / len(loader)
