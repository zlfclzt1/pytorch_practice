import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image

from utils.backbone import Resnet50

if __name__ == "__main__":
    image = Image.open('/home/zlf/sshfs/3090/media/gpu/sdf/result/240229_150_6v1l_k/heatmaps/hjdataset_release_test_v1.0/223.png')



    batch_size, h, w = 3, 256, 704
    backbone = Resnet50()
    optimizer = optim.Adam(backbone.parameters(), lr=0.01)
    bceloss = nn.BCELoss()

    for step in range(50):
        backbone.train()
        optimizer.zero_grad()
        torch.set_grad_enabled(True)

        input = torch.randn((batch_size, 3, h, w))
        out_features = backbone(input)
        result = out_features[-1]
        result = F.sigmoid(result)

        target = torch.randn(result.shape, device=result.device)
        loss = bceloss(result, target)
        loss.backward()
        optimizer.step()

