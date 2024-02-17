import torch
import torch.nn as nn

import torchvision.models as models

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()

        self.resnet50 = list(models.resnet50().children())[:-2]

    def forward(self, input):
        x = input
        for idx, layer in enumerate(self.resnet50):
            x = layer(x)
        return x


if __name__ == "__main__":
    resnet50 = Resnet50()
    input = torch.randn((8, 3, 256, 704))

    result = resnet50(input)
