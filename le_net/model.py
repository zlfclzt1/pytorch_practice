import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from einops import rearrange

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=1, padding=0)

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, input):
        # input B * 1 * 32 * 32
        x = self.conv1(input) # B * 6 * 28 * 28
        x = F.max_pool2d(x, kernel_size=(2,2), stride=2) # B * 6 * 14 * 14
        x = self.conv2(x) # B * 16 * 10 * 10
        x = F.max_pool2d(x, kernel_size=(2,2), stride=2) # B * 16 * 5 * 5

        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



if __name__ == "__main__":
    b, c, h, w = 8, 1, 32, 32
    le_net = LeNet()
    optimizer = optim.Adam(le_net.parameters())

    for step in range(1000):
        print(step)
        optimizer.zero_grad()

        input = torch.rand((b, c, h, w))
        output = le_net(input)
        target = torch.rand(b, 10)

        mse_loss = nn.MSELoss()
        loss = mse_loss(output, target)
        loss.backward()

        optimizer.step()
