import torch
import torch.nn.functional as F

from pytorch_toolbelt.losses import DiceLoss, SoftBCEWithLogitsLoss


class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.dice = DiceLoss(mode="binary", from_logits=False)
        self.bce = torch.nn.BCELoss()

    def forward(self, inputs, targets, smooth=0.0):
        dice = self.dice(inputs, targets)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        bce = self.bce(inputs.float(), targets.float())

        div = 1 - (torch.sum(inputs) / (torch.sum(targets) + 0.0000001))
        if div < 0:
            div = 1

        loss = bce + div

        return loss, dice, bce


class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, use_sigmoid=False, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        if use_sigmoid:
            inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky
