from graphs.losses import bce,focal,wbce,dice,l1

def define_loss( config, device, reduction='mean'):
    loss = config.loss

    def criterion(l, device):
        l = l.upper()
        if l == "CROSSENTROPY" or l == "BCE":
            return bce.BCEWithLogitsLoss(reduction=reduction).to(device)
        elif l == "FOCAL":
            return focal.FocalLoss(reduction=reduction).to(device)
        elif l == "DICE":
            return dice.DiceLoss(reduction=reduction).to(device)
        elif l == "WBCE":
            return wbce.WeightedBCEWithlogits(config.WBCE_weights, device=device)
        elif l == "L1":
            return l1.L1Loss().to(device)
        else:
            raise ValueError("Unknow loss function : " + loss)

    return [criterion(l, device) for l in loss]

def compute_loss(loss,output,target,weights):
    loss=loss
    i=0
    for l,w in zip(loss,weights):
        if i==0 :
            s=l(output,target)*w
            i+=1
        else:
            s+=l(output,target)*w
    return s