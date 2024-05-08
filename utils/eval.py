
import torch
from utils.config import HEIGHT, WIDTH, N_VAR
import wandb


def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)


def latitude_weighting_factor(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s


def weighted_rmse_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each channel
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1, -2)))
    return result


def weighted_acc_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc for each channel
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1, -2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1, -2)) * torch.sum(weight * target *
                                                                                                                                    target, dim=(-1, -2)))
    return result


def evaluate(model, loader, device, subset=None, autoreg=True, log=True):
    model.eval()  # Set the model to evaluation mode
    total_rmse = 0.0
    total_acc = 0.0
    total_batches = len(loader) if subset is None else subset
    with torch.no_grad():  # Disable gradient computation
        for batch_index, data in enumerate(loader):
            if subset is not None and batch_index >= subset:
                break

            x, edge_index = data
            x = x.to(device)
            edge_index = edge_index.to(device).squeeze()

            batch_size, seq_len, num_nodes, num_features = x.shape

            # Initialize containers for metrics
            acc_per_pred_step = torch.zeros(seq_len - 1).to(device)
            rmse_per_pred_step = torch.zeros(seq_len - 1).to(device)

            # Iterate over prediction steps
            for t in range(seq_len - 1):
                if autoreg and t > 0:
                    x_input = predictions.view(batch_size, num_nodes, num_features)
                else:
                    x_input = x[:, t, :, :].view(batch_size, num_nodes, num_features)

                x_target = x[:, t + 1, :, :].view(batch_size, num_nodes, num_features)

                predictions = model(x_input, edge_index)

                x_target = x_target.view(1, N_VAR, HEIGHT, WIDTH)
                predictions = predictions.view(1, N_VAR, HEIGHT, WIDTH)
                # Compute the RMSE and accuracy for each prediction step
                rmse = weighted_rmse_channels(predictions, x_target)
                acc = weighted_acc_channels(predictions, x_target)

                acc_per_pred_step[t] += acc.mean().item()
                rmse_per_pred_step[t] += rmse.mean().item()

            # Accumulate total metrics
            total_rmse += rmse_per_pred_step.mean().item()  # Sum over all prediction steps
            total_acc += acc_per_pred_step.mean().item()
            # Log after every 100 batches
            if batch_index % 100 == 0:
                print(f"Batch {batch_index + 1}/{total_batches} - RMSE: {rmse_per_pred_step.mean():0.3f}, Accuracy: {acc_per_pred_step.mean():0.3f}")

    if subset is not None:
        total_batches = max(total_batches, subset)

    avg_rmse = total_rmse / (total_batches)
    avg_acc = total_acc / (total_batches)
    print(f"Average RMSE: {avg_rmse:.3f}, Average Accuracy: {avg_acc:.3f}")

    acc_per_pred_step /= total_batches
    rmse_per_pred_step /= total_batches
    print(f"Average RMSE per prediction step: {rmse_per_pred_step}")
    print(f"Average Accuracy per prediction step: {acc_per_pred_step}")

    if log:

        wandb.log({"val_rmse": avg_rmse, "val_acc": avg_acc})
        # log histogram of acc_per_pred_step
        # log acc tensor to plot later
        log_values_over_time(acc_per_pred_step, name="Accuracy")
        log_values_over_time(rmse_per_pred_step, name="RMSE")

    return avg_rmse, avg_acc


def log_values_over_time(values, name="Accuracy", time_step=6, single_plot=True):
    """
    Log values for multiple future time steps.

    Args:
    values (list or tensor): A list or tensor containing values for future time steps.

    """
    # Create a dictionary to log
    if single_plot:
        log_data = {}
        for i, acc in enumerate(values):
            # one step is 6 h
            days = (i + 1) * time_step / 24
            log_data[f"Forecast {name} after {days} days"] = acc
        # x axis in days
        wandb.log(log_data)
    else:
        for i, acc in enumerate(values):
            # one step is 6 h
            days = (i + 1) * time_step / 24
            wandb.log({f"Forecast {name} after days": acc + (n+1)*0.1}, step=i)
