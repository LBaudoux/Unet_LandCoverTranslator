import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#class DiceLoss(nn.Module):
#
#   def __init__(self,smooth=1.):
#        super(DiceLoss, self).__init__()
#        self.smooth=smooth
#
#    def forward(self,pred,targets):
#        pred=F.softmax(pred, dim=1)
#
#        intersection = (pred * targets).sum(dim=(0,2,3))
#
#        loss = (1 - ((2. * intersection ) / (pred + targets).sum(dim=(0,2,3))))
#        return loss.mean()

class DiceLoss(nn.Module):

    def __init__(self,smooth=1.,reduction="mean"):
        super(DiceLoss, self).__init__()
        self.smooth=smooth
        self.reduction=reduction

    def forward(self,pred,targets):
        pred=F.softmax(pred, dim=1)

        intersection = (pred * targets).sum(dim=(2,3))

        loss = (1 - ((2. * intersection ) / (pred.sum(dim=(2,3)) + targets.sum(dim=(2,3)))))
        if self.reduction=="mean":
            return loss.mean()
        elif self.reduction=='none':
            return loss.mean(dim=1)
        elif self.reduction=='sum':
            return loss.sum()
        else :
            raise ValueError("Unknown reduction parameter : {}".format(self.reduction))