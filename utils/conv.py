import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, requires_grad=True))
        # self.bias = nn.Parameter(torch.randn(1))

    def forward(self, inputs):
        # inputs B * C * H * W
        B, C, H, W = inputs.shape
        device = inputs.device
        assert C == self.in_channels

        #padding
        padded_input = torch.zeros((B, C, H+2*self.padding, W+2*self.padding), device=device)
        padded_input[:, :, self.padding:-self.padding, self.padding:-self.padding] = inputs

        # padded_input = torch.cat([torch.zeros(B, C, self.padding, W), inputs,
        #                           torch.zeros(B, C, self.padding, W)], dim=2)
        # padded_input = torch.cat([torch.zeros(B, C, H+self.padding*2, self.padding), padded_input,
        #                           torch.zeros(B, C, H+self.padding*2, self.padding)], dim=-1)

        # padded_input = F.pad(inputs, [self.padding, self.padding, self.padding, self.padding], value=0)

        result_h = (H + self.padding * 2 - self.kernel_size)//self.stride + 1
        result_w = (W + self.padding * 2 - self.kernel_size)//self.stride + 1
        result = torch.zeros((B, self.out_channels, result_h, result_w), device=device)

        for b in range(B):
            for i in range(result_h):
                for j in range(result_w):
                    cur_input = padded_input[b, :, i:i + self.kernel_size, j:j + self.kernel_size]

                    result[b, :, i, j] = (cur_input[None, ...] * self.weight).sum([1, 2, 3])
        # tmp = F.conv2d(inputs, self.weight, padding=self.padding)
        return result

if __name__ == '__main__':
    con2d = Conv2d(16, 8, [3, 3], stride=1, padding=1)
    input_tensor = torch.randn(4, 16, 5, 5)
    con2d.forward(input_tensor)
