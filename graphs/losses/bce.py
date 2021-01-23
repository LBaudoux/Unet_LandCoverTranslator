"""
An example for loss class definition, that will be used in the agent
"""
import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):
    def __init__(self,reduction='mean'):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.reduction =reduction

    def forward(self, logits, labels):
        if self.reduction=='mean' or self.reduction=='sum':
            return self.loss(logits.float(), labels.float())
        elif self.reduction=='none':
            return self.loss(logits.float(), labels.float()).mean(dim=(1,2,3))
        else :
            raise ValueError("Unknown reduction parameter : {}".format(self.reduction))