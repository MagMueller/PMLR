from torch import nn
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import einops
# # this modified for 2D images as input


class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, kernel_size ,padding, ** kwargs):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        # self.i2h = PixelwiseConv2d(n_inp + n_hid + n_hid, n_hid, 3, 12, 10, stride=1, padding=1, dilation=1)
        self.i2h = nn.Conv2d(n_inp + n_hid + n_hid, n_hid, kernel_size, stride=1, padding=padding, dilation=1,
                             padding_mode="replicate")

    def forward(self, x, hy, hz):
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((x, hz, hy), 1)))
                             - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz


class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, dt, gamma, epsilon, kernel_size, **kwargs):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kernel size should be a positive odd number")
        padding = (kernel_size - 1) // 2
        self.cell = coRNNCell(n_inp, n_hid, dt, gamma, epsilon, kernel_size, padding)
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
    def __init__(self, n_hid, dt, gamma, epsilon, kernel_size, padding, **kwargs):
        super(coRNNCell2, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = PixelwiseConv2d(n_hid + n_hid, n_hid, kernel_size, 12, 10, stride=1, padding=padding, dilation=1)
        # self.i2h = nn.Conv2d(n_hid + n_hid, n_hid, kernel_size, stride=1, padding=1, dilation=1, 
        #                     padding_mode="replicate")

    def forward(self, hy, hz):
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((hz, hy), 1)))
                             - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz


class coRNN2(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, dt, gamma, epsilon, seq_len, n_roll, kernel_size, **kwargs):
        super(coRNN2, self).__init__()
        self.n_hid = n_hid
        self.n_roll = n_roll

        # TODO: dt inverse of n_roll
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kernel size should be a positive odd number")
        padding = (kernel_size - 1) // 2

        self.readin = nn.Conv2d(seq_len*n_inp, 2*n_hid, kernel_size, stride=1, padding=padding, dilation=1,
                                padding_mode="replicate")

        self.cell = coRNNCell2(n_hid, dt, gamma, epsilon, kernel_size, padding)

        self.readout = nn.Conv2d(n_hid, n_out, 1)

    def forward(self, x):  # (T,B,C,H,W)
        # initialize hidden states
        # print(x.shape)
        x = einops.rearrange(x, 't b c h w -> b (t c) h w')  # (B,TC,H,W)
        # x = x.transpose(0, 1).reshape(x.shape[1], x.shape[0]*x.shape[2], x.shape[3], x.shape[4])  # (B,TC,H,W)
        h = self.readin(x)  # (B,seq_len*C,H,W)
        hy, hz = h[:, :self.n_hid], h[:, self.n_hid:]

        for t in range(self.n_roll):
            hy, hz = self.cell(hy, hz)  # (B,C,H,W)
        output = self.readout(hy)
        return output  # (B,C',H,W)


class PixelwiseConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, input_height, input_width, stride=1, padding=0, dilation=1, batch_size=30):
        super(PixelwiseConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size

        # Calculate the output dimensions
        self.output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        self.output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        # Define a weight tensor for each pixel in the output
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, self.output_height, self.output_width))
        self.bias = nn.Parameter(torch.randn(out_channels, self.output_height, self.output_width))

    def forward(self, x):
   
        batch_size = x.size(0)

        # Unfold the input into sliding local blocks
        x_unfolded = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)
        #print(x_unfolded.shape)

        # Reshape unfolded input to (batch_size, in_channels, kernel_size, kernel_size, output_height, output_width)
        x_unfolded = x_unfolded.view(batch_size, self.in_channels, self.kernel_size, self.kernel_size, self.output_height, self.output_width)
        #print(x_unfolded.shape)
        #print(self.weights.shape)

        # Perform element-wise multiplication and sum over in_channels, kernel_size, and kernel_size
        output = (x_unfolded.unsqueeze(1) * self.weights).sum(dim=(2, 3, 4)) + self.bias
        #print(output.shape)
        #print()

        return output