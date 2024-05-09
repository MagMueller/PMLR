
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


def evaluate(model, loader, device, subset=None, autoreg=True, log=True, prediction_len=10):
    if model is not None:
        model.eval()  # Set the model to evaluation mode
    total_rmse = 0.0
    total_acc = 0.0
    total_batches = len(loader) if subset is None else subset
    print(total_batches)
    counter = -1
    prediction_len = min(prediction_len, len(loader))
    acc_per_pred_step = torch.zeros(prediction_len).to(device)
    rmse_per_pred_step = torch.zeros(prediction_len).to(device)
    n_times = torch.zeros(prediction_len).to(device)
    total = total_batches / prediction_len
    print(f"Total sequences to evaluate: {total} with prediction_len: {prediction_len} ")
    counter2 = 0
    with torch.no_grad():  #
        # assume sorted loader
        for batch_index, data in enumerate(loader):
            counter += 1
            if counter == prediction_len:
                counter = 0
                counter2 += 1
                print(f"Seq {counter2}/{total} - RMSE: {rmse_per_pred_step.mean():0.3f}, Accuracy: {acc_per_pred_step.mean():0.3f}")
                if subset is not None and counter2 >= subset:
                    break
            if subset is not None and batch_index >= subset:
                break

            x, edge_index = data
            x = x.to(device)
            edge_index = edge_index.to(device).squeeze()
            batch_size, seq_len, num_nodes, num_features = x.shape
            if counter == 0:
                x_input = x[:, 0, :, :].view(batch_size, num_nodes, num_features)
            else:
                x_input = predictions.view(batch_size, num_nodes, num_features)

            x_target = x[:, 1, :, :].view(batch_size, num_nodes, num_features)
            if model is not None:
                predictions = model(x_input, edge_index)
            else:
                predictions = x_input

            x_target = x_target.view(1, N_VAR, HEIGHT, WIDTH)
            predictions = predictions.view(1, N_VAR, HEIGHT, WIDTH)
            # Compute the RMSE and accuracy for each prediction step
            rmse = weighted_rmse_channels(predictions, x_target)
            acc = weighted_acc_channels(predictions, x_target)

            acc_per_pred_step[counter] += acc.mean().item()
            rmse_per_pred_step[counter] += rmse.mean().item()
            n_times[counter] += 1
            # Log after every 100 batches
            # if batch_index % 10 == 0:
            #     print(f"Batch {batch_index + 1}/{total_batches} - RMSE: {rmse_per_pred_step.mean():0.3f}, Accuracy: {acc_per_pred_step.mean():0.3f}")

    if subset is not None:
        total_batches = min(total_batches, subset)

    # Accumulate total metrics
    # if n_times is 0 cut acc_per_pred_step and rmse_per_pred_step else devide with value
    mask = n_times == 0
    n_times[mask] = 1
    acc_per_pred_step /= n_times
    rmse_per_pred_step /= n_times
    # cut values where n_times is 0
    acc_per_pred_step = acc_per_pred_step[~mask]
    rmse_per_pred_step = rmse_per_pred_step[~mask]

    avg_rmse = rmse_per_pred_step.mean().item()  # Sum over all prediction steps
    avg_acc = acc_per_pred_step.mean().item()
    print(f"Average RMSE: {avg_rmse:.3f}, Average Accuracy: {avg_acc:.3f} over {total} seq and prediction_len {prediction_len}.")

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
            wandb.log({f"Forecast {name} after days": acc}, step=i)
