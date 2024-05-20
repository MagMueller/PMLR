
import torch


def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)


def latitude_weighting_factor(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s


def weighted_rmse_channels(pred: torch.Tensor, target: torch.Tensor, n_var=20, height=721, width=1440) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each channel
    # TODO: check if this is correct - else reshape
    # reshape input [batch, h*w, c] -> [batch, c, h, w]
    pred = pred.permute(0, 2, 1).reshape(-1, n_var, height, width)
    target = target.permute(0, 2, 1).reshape(-1, n_var, height, width)
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1, -2)))
    return result


def weighted_acc_channels(pred: torch.Tensor, target: torch.Tensor, n_var=20, height=721, width=1440) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc for each channel
    # reshape input [batch, h*w, c] -> [batch, c, h, w]
    pred = pred.permute(0, 2, 1).reshape(-1, n_var, height, width)
    target = target.permute(0, 2, 1).reshape(-1, n_var, height, width)

    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1, -2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1, -2)) * torch.sum(weight * target *
                                                                                                                                    target, dim=(-1, -2)))
    return result
