import torch.nn as nn

class Relu(nn.Module):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x):
        mask = x < 0
        x[mask] = 0.0
        return x