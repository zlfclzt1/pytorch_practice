import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn((out_features, in_features)))
        self.bias = nn.Parameter(torch.randn(out_features, requires_grad=True))

    def forward(self, x):
        y = torch.matmul(self.weight, x.T).T + self.bias
        return y

if __name__ == "__main__":
    input = torch.randn((3,4))
    linear = Linear(in_features=4, out_features=2)
    output = linear(input)
