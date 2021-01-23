import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class WeightedBCEWithlogits(nn.Module):
    def __init__(self, effectif,device,shape=36,N_0=100,r=0.1):
        super(WeightedBCEWithlogits, self).__init__()
        prop = torch.tensor(effectif,device=device)
        theo_prop = prop.float() / prop.sum()
        self.weight=N_0*torch.exp(-theo_prop/r)
        self.weight = self.weight.repeat_interleave(shape**2)

    def forward(self, output, target):
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        loss = criterion(output, target)
        _,n,x,y=target.shape
        return (loss * self.weight.reshape(n,x,y)).mean()