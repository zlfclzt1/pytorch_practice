import torch
import torch.nn as nn


class SoftMax(nn.Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def forward(self, x):
        exp_x = torch.exp(x)
        exp_sum = exp_x.sum(dim=-1, keepdim=True)

        output = exp_x / exp_sum
        return output

if __name__ == "__main__":
    input = torch.randn((2, 4, 3))
    soft_max = SoftMax()
    output = soft_max(input)