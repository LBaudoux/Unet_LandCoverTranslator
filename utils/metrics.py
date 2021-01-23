"""
This file will contain the metrics of the framework
"""
import numpy as np
import torch
import torch.nn.functional as F


class EdgePreservationAssessment(object):

    def __init__(self, outputs, targets):
        with torch.no_grad():
            if outputs.shape[1] > 1:
                self.pred = torch.argmax(outputs, 1, keepdim=True).double()
                self.targ = torch.argmax(targets, 1, keepdim=True).double()
            else:
                self.pred = outputs
                self.targ = targets

    def laplace_filter(self, image):
        with torch.no_grad():
            filter = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=image.device).double()
            filter = filter.expand(1, 1, 3, 3)
            hp = F.conv2d(image.double(), filter.double(), padding=0)
            return torch.where(hp != 0, torch.tensor([1]), torch.tensor([0])).float()

    def EPI(self):
        with torch.no_grad():

            if torch.isnan(self.pred).sum() != 0:
                raise ValueError("Le arg max pred merde")

            hp_pred = self.laplace_filter(self.pred)

            if torch.isnan(hp_pred).sum() != 0:
                raise ValueError("Le filtre de laplace sur pred merde")

            if torch.isnan(self.targ).sum() != 0:
                raise ValueError("Le arg max targ merde")

            hp_targ = self.laplace_filter(self.targ)

            if torch.isnan(hp_targ).sum() != 0:
                raise ValueError("Le filtre de laplace sur targ merde")

            ec_tgt = hp_targ - torch.mean(hp_targ, dim=(2, 3), keepdim=True)
            ec_src = hp_pred - torch.mean(hp_pred, dim=(2, 3), keepdim=True)

            if torch.isnan(ec_tgt).sum() != 0 or torch.isnan(ec_src).sum() != 0:
                raise ValueError("Ec merde")

            num = torch.sum(ec_tgt * ec_src, dim=(1, 2, 3))
            denom1 = torch.sum(ec_tgt * ec_tgt, dim=(1, 2, 3))
            denom2 = torch.sum(ec_src * ec_src, dim=(1, 2, 3))
            denom = torch.sqrt(denom1 * denom2)
            if torch.isnan(denom).sum() != 0:
                raise ValueError("denom merde")

            EPI = torch.mean(num / (denom + 0.00000001))

            return EPI


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc..
    """

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg


class AverageMeterList:
    """
    Class to be an average meter for any average metric List structure like mean_iou_per_class
    """

    def __init__(self, num_cls):
        self.cls = num_cls
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls
        self.reset()

    def reset(self):
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls

    def update(self, val, n=1):
        for i in range(self.cls):
            self.value[i] = val[i]
            self.sum[i] += val[i] * n
            self.count[i] += n
            self.avg[i] = self.sum[i] / self.count[i]

    @property
    def val(self):
        return self.avg


def cls_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k / batch_size)
    return res
