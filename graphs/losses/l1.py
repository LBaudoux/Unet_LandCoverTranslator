import torch.nn as nn


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss