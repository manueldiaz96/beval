"""
File containing various functions different for Losses used

This file contains code from https://github.com/nv-tlabs/lift-splat-shoot
License available in https://github.com/nv-tlabs/lift-splat-shoot/blob/master/LICENSE

This file contains code from https://github.com/wayveai/fiery/tree/master/fiery
License available at https://github.com/wayveai/fiery/blob/master/LICENSE

"""

import torch
import torch.nn.functional as F


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss


class BCEMultiClass(torch.nn.Module):
    def __init__(self, weights):
        super(BCEMultiClass, self).__init__()
        self.weights = weights
        self.num_clases = weights.shape[0]
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, ypred, ytgt):
        loss = 0
        for i in range(self.num_clases):
            self.loss_fn.pos_weight = self.weights[i]
            binimgs_cls = (ytgt == i).float()
            loss += self.loss_fn(ypred[:, i], binimgs_cls)
        return loss


class DiceLoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, ypred, ytgt):
        loss = 0
        for i in range(self.num_classes):
            target = (ytgt == i).float()
            input = ypred[:, i].sigmoid()

            input = input.contiguous().view(input.size()[0], -1)
            target = target.contiguous().view(target.size()[0], -1).float()

            a = torch.sum(input * target, 1)
            b = torch.sum(input * input, 1) + 0.001
            c = torch.sum(target * target, 1) + 0.001
            d = (2 * a) / (b + c)
            loss += (1-d).mean()

        return loss


def cumsum_trick(x, geom_feats, ranks):
    # breakpoint()
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        #breakpoint()
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None

def batch_soft_iou(preds, binimgs):
    
    preds = preds.sigmoid()
    tgt = binimgs
    
    num = preds * tgt
    den = preds + tgt - num

    iou = num.sum() / den.sum()

    return iou 

class SpatialRegressionLoss(torch.nn.Module):
    def __init__(self, norm, ignore_index=255, future_discount=1.0):
        super(SpatialRegressionLoss, self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        assert len(prediction.shape) == 4, 'Must be a 5D tensor'
        # ignore_index is the same across all channels
        mask = target[:, :1] != self.ignore_index
        if mask.sum() == 0:
            return prediction.new_zeros(1)[0].float()

        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=-3, keepdims=True)

        return loss[mask].mean()
    
    
def compile_loss(num_classes=1, add_map=False, gpuid=0, task='semseg', train_label=None, pos_weight=2.14):

    assert task in ['semseg', 'instseg'], "Task must be semseg or instseg"

    if num_classes == 1 and not add_map:
        sem_loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    else:
        sem_loss_fn = torch.nn.CrossEntropyLoss().cuda(gpuid)

        if num_classes == 7:
            weights = torch.tensor([0.2210,    # Unknown
                                    178.5110,  # Human
                                    100.7730,  # Movable Object
                                    6.2840,    # Vehicle
                                    0.6410,    # Drivable Area
                                    1.7620,    # Walkway
                                    5.8110     # Lane-divider
                                    ]).cuda(gpuid)

            text_labels = [
                'Unknown',
                'Human',
                'Movable_Object',
                'Vehicle',
                'Drivable_Area',
                'Walkway',
                'Lane_Divider',
            ]
            sem_loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        elif num_classes==4 :
            text_labels = [
                'Unknown',
                train_label.capitalize(),
                'Drivable_Area',
                'Walkway'
            ]
            #                             Unknown  class    driv_a  wlkway
            weights = {'vehicle':        [0.3860,  10.9920, 1.0080, 3.0470],
                       'human':          [0.3830, 309.5550, 0.9450, 3.0260],
                       'movable_object': [0.3840, 175.3900, 0.9470, 3.0230]}[train_label]

            
            
        elif num_classes==3 :
            text_labels = [
                'Unknown',
                train_label.capitalize(),
                'Drivable_Area'
            ]
            weights = [0.3860,  10.9920, 1.0080]
        
        weights = torch.tensor(weights).cuda(gpuid)

        sem_loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            

    
    if task == 'semseg':
        if num_classes > 1:
            return sem_loss_fn, text_labels
        else:
            return sem_loss_fn
    
    else:
        assert "Task not supported"