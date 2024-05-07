
import torch
from utils.config import HEIGHT, WIDTH, N_VAR


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


def evaluate(model, loader, device, subset=None, autorereg=False):
    model.eval()  # Set the model to evaluation mode
    total_rmse = 0
    total_acc = 0
    total_batches = len(loader) if subset is None else subset

    with torch.no_grad():  # Disable gradient computation
        for batch, data in enumerate(loader):
            if subset is not None and batch >= subset:
                break
            x, edge_index = data
            # Shape: [batch_size, sequence_length, num_nodes, num_features]
            x = x.to(device)
            edge_index = edge_index.to(device).squeeze()


            batch_size, seq_len, num_nodes, num_features = x.shape
            predictions = None
            acc_per_pred_step = []
            rmse_per_pred_step = []
            # Iterate over each timestep, predicting the next state except for the last since no next state exists
            for t in range(seq_len - 1):
                if autorereg:
                    if predictions is None:
                        x_input = x[:, t, :, :].view(batch_size, num_nodes, num_features)
                    else:
                        x_input = predictions.view(1, num_nodes, num_features)
                else:
                    x_input = x[:, t, :, :].view(batch_size, num_nodes, num_features)

                x_target = x[:, t + 1, :, :].view(batch_size, num_nodes, num_features)

                # Model output for current timestep
                predictions = model(x_input, edge_index)
                # reshape to [1, 21, height, width]
                x_target = x_target.view(1, N_VAR, HEIGHT, WIDTH)
                predictions = predictions.view(1, N_VAR, HEIGHT, WIDTH)
                predictions = predictions
                # Calculate the latitude-weighted RMSE and accuracy for the predictions
                rmse = weighted_rmse_channels(predictions, x_target)
                acc = weighted_acc_channels(predictions, x_target)

                # sum
                rmse = torch.mean(rmse)
                acc = torch.mean(acc)
                total_rmse += rmse.item()
                total_acc += acc.item()
                # acc_per_pred_step.append(acc.item())
                # rmse_per_pred_step.append(rmse.item())


                # print(f"RMSE: {rmse.item()}, Accuracy: {acc.item()}")
            if batch % 100 == 0:
                print(f"Batch {batch + 1}/{total_batches} - RMSE: {rmse.item():0.3f}, Accuracy: {acc.item():0.3f}")



    avg_rmse = total_rmse / total_batches / (seq_len - 1)
    avg_acc = total_acc / total_batches / (seq_len - 1)
    print(f"Average RMSE: {avg_rmse:.3f}, Average Accuracy: {avg_acc:.3f}")
    return avg_rmse, avg_acc
