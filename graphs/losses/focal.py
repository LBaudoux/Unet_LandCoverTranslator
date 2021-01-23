import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
            
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction=="mean":
            return torch.mean(F_loss)
        elif self.reduction == 'none':
            return F_loss
        elif self.reduction == 'sum':
            return F_loss.sum()
        else :
            raise ValueError("Unknown reduction parameter : {}".format(self.reduction))