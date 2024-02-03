import torch
import torch.nn as nn
import torch.nn.functional as F





class Detr(nn.Module):
    def __init__(self):
        super(Detr, self).__init__()

        self.backbone = Backbone()

    def forward(self, input):
        x = self.backbone(input)

        return x


if __name__ == "__main__":
    detr = Detr()
    images = torch.randn((2, 3, 256, 704))

    result = detr(images)