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


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        self.resnet50 = models.resnet50()

    def forward(self, input):
        output = []
        x = input
        for name, sub_model in list(self.resnet50.named_children()):
            print(name)
            x = sub_model(x)



if __name__ == "__main__":
    input = torch.randn((2, 3, 704, 256))
    fpn = FPN()
    result = fpn(input)


