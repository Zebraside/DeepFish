import torch
import torch.nn.functional as F

from pytorch_toolbelt.losses import DiceLoss, SoftBCEWithLogitsLoss


class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.dice = DiceLoss(mode="binary", from_logits=True)
        self.bce = F.binary_cross_entropy_with_logits

    def forward(self, inputs, targets, smooth=0.0):
        dice = self.dice(inputs, targets)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        bce = self.bce(inputs, targets)

        loss = bce + dice

        return loss


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


class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy_with_logits(inputs, targets)
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss