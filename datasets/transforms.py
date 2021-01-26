import torch
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return [[torch.from_numpy(i).float() for i in sample[0]],[torch.from_numpy(i).float() for i in sample[1]]]

class ToOneHot(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,nclasses,idx1=1,idx2=0):
        self.nclasses=nclasses+1
        self.idx1=idx1
        self.idx2=idx2

    def __call__(self, sample):
        # oso_one_hot=OneHotEncoder(categories=range(1,23))
        with torch.no_grad():
            # print(torch.max(sample["oso"].long()))
            sample[self.idx1][self.idx2]=F.one_hot(sample[self.idx1][self.idx2].long(), num_classes=self.nclasses).permute(0,3,1,2,)[0][1:]
        # oso_one_hot = OneHotEncoder(categories=[111,112,121,122,123,124])
        # sample["oso_enc"] = oso_one_hot.transform(sample["oso"])
        return sample