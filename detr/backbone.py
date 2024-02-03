import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        resnet50 = models.resnet50()
        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-2])
    def forward(self, input):
        return self.resnet50(input)


