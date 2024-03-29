import torch
import torch.nn as nn


class BatchNorm2D(nn.Module):
    def __init__(self, num_features, eps=1e-6, momentum=0.1):
        super(BatchNorm2D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(self.num_features, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(self.num_features, requires_grad=True))

        self.first = True
        self.running_mean = torch.zeros(self.num_features)
        self.running_var = torch.zeros(self.num_features)

    def forward(self, x, train_mode=True):
        assert len(x.shape) == 4  # NCHW

        if train_mode:
            mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
            var = torch.var(x, dim=[0, 2, 3], unbiased=False, keepdim=True)
        else:
            mean = self.running_mean
            var = self.running_var

        out = (x - mean) / (torch.sqrt(var) + self.eps)
        out = out * self.gamma[None, :, None, None] + self.beta[None, :, None, None]

        if train_mode:
            if self.first:
                self.running_mean = mean
                self.running_var = var
            else:
                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean
                self.running_var =  (1-self.momentum) * self.running_var + self.momentum * var
        return out

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = normalized_shape
        self.gamma = nn.Parameter(torch.randn(normalized_shape))
        self.beta = nn.Parameter(torch.randn(normalized_shape))

    def forward(self, input):
        n = len(self.normalized_shape)
        process_dim = torch.arange(-n, 0).tolist()

        mean = torch.mean(input, dim=process_dim, keepdim=True)
        var = torch.var(input, dim=process_dim, unbiased=False, keepdim=True)
        x = (input - mean) / torch.sqrt(var + 1e-6)

if __name__ == "__main__":
    bn = BatchNorm2D(8)
    input = torch.randn((4, 8, 16, 16), dtype=torch.float32)
    out = bn(input)

    input1 = torch.randn((2, 8, 16))
    ln = LayerNorm(normalized_shape=[8, 16])

    result = ln(input1)
