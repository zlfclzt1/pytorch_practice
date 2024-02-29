import torch
import torch.nn as nn

import torchvision.models as models
from collections import OrderedDict


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()

        self.resnet50 = nn.ModuleList(list(models.resnet50().children())[:-2])
        self.out_channels = [256, 512, 1024, 2048]

    def forward(self, x):
        output = []

        for idx, layer in enumerate(self.resnet50):
            x = layer(x)
            if idx > 3:
                output.append(x)
        return output

if __name__ == "__main__":
    resnet50 = Resnet50()
    input = torch.randn((8, 3, 256, 704))

    result = resnet50(input)
