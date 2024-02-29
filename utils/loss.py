import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred, target):
        b, c, h, w = target.shape
        num = b * h * w

        pred = F.softmax(pred, dim=1)
        pred_log = torch.log(pred)
        loss = -(pred_log * target.type(torch.float32)).sum() / num
        return loss


# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          gamma=2.0,
                          alpha=0.25):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred).view(-1, 1)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, beta=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.beta = beta

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)

        # pt =

        # pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        # focal_weight =


        log_pt = torch.log(pt)

        # loss = (-1 * foreback_weight * (1 - pt)**self.gamma * log_pt).sum()
        return loss


if __name__ == "__main__":
    b, c, h, w = 3, 2, 64, 32
    num = b * h * w
    pred = torch.randn((b * h * w, c), dtype=torch.float32)
    # pred = F.softmax(pred, dim=1)

    label = torch.randint(low=0, high=c, size=(num, )).reshape(-1)
    target = F.one_hot(label, num_classes=c)

    focal_loss = FocalLoss()
    cross_entropy_loss = CrossEntropyLoss()
    cross_entropy_loss_ = nn.CrossEntropyLoss()

    loss = focal_loss(pred, label)
    loss_ = py_sigmoid_focal_loss(pred, label)


    # loss = cross_entropy_loss(pred, target)
    # loss_ = cross_entropy_loss_(pred, target)
