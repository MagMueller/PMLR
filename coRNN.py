from torch import nn
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import einops
# # this modified for 2D images as input


class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, **kwargs):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        # self.i2h = PixelwiseConv2d(n_inp + n_hid + n_hid, n_hid, 3, 12, 10, stride=1, padding=1, dilation=1)
        self.i2h = nn.Conv2d(n_inp + n_hid + n_hid, n_hid, 3, stride=1, padding=1, dilation=1,
                             padding_mode="replicate")

    def forward(self, x, hy, hz):
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((x, hz, hy), 1)))
                             - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz


class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, dt, gamma, epsilon, **kwargs):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.cell = coRNNCell(n_inp, n_hid, dt, gamma, epsilon)
        self.readout = nn.Conv2d(n_hid, n_out, 1)

    def forward(self, x):  # (T,B,C,H,W)
        # initialize hidden states
        hy = torch.zeros(x.shape[1], self.n_hid, x.shape[3], x.shape[4], device=x.device, dtype=x.dtype)  # (B,C,H,W)
        hz = torch.zeros(x.shape[1], self.n_hid, x.shape[3], x.shape[4], device=x.device, dtype=x.dtype)
        for t in range(x.shape[0]):
            hy, hz = self.cell(x[t], hy, hz)  # (B,C,H,W)
        output = self.readout(hy)
        return output  # (B,C',H,W)


class coRNNCell2(nn.Module):
    def __init__(self, n_hid, dt, gamma, epsilon):
        super(coRNNCell2, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        # self.i2h = PixelwiseConv2d(n_hid + n_hid, n_hid, 3, 12, 10, stride=1, padding=1, dilation=1)
        self.i2h = nn.Conv2d(n_hid + n_hid, n_hid, 3, stride=1, padding=1, dilation=1,
                             padding_mode="replicate")

    def forward(self, hy, hz):
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((hz, hy), 1)))
                             - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz


class coRNN2(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, dt, gamma, epsilon, seq_len, n_roll, **kwargs):
        super(coRNN2, self).__init__()
        self.n_hid = n_hid
        self.n_roll = n_roll
        self.readin = nn.Conv2d(seq_len*n_inp, 2*n_hid, 3, stride=1, padding=1, dilation=1,
                                padding_mode="replicate")
        self.cell = coRNNCell2(n_hid, dt, gamma, epsilon)
        self.readout = nn.Conv2d(n_hid, n_out, 1)

    def forward(self, x):  # (T,B,C,H,W)
        # initialize hidden states
        print(x.shape)
        x = einops.rearrange(x, 't b c h w -> b (t c) h w')  # (B,TC,H,W)
        # x = x.transpose(0, 1).reshape(x.shape[1], x.shape[0]*x.shape[2], x.shape[3], x.shape[4])  # (B,TC,H,W)
        print(f"shape after reshape: {x.shape}")
        h = self.readin(x)  # (B,seq_len*C,H,W)
        print(f"shape after readin: {h.shape}")
        hy, hz = h[:, :self.n_hid], h[:, self.n_hid:]
        print(f"shape after readin hy: {hy.shape} hz: {hz.shape}")
        for t in range(self.n_roll):
            hy, hz = self.cell(hy, hz)  # (B,C,H,W)
        output = self.readout(hy)
        return output  # (B,C',H,W)
