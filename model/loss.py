import torch
import torch.nn as nn
import torch.nn.functional as F
from .diou_loss import DIoULoss

def focal_loss(pred, target):
    """
    classifier loss of focal loss
    Args:
        pred: heatmap of prediction
        target: heatmap of ground truth

    Returns: cls loss

    """
    # Find every image positive points and negative points,
    # one bounding box corresponds to one positive point,
    # except positive points, other feature points are negative sample.

    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    

    # The negative samples near the positive sample feature point have smaller weights
    neg_weights = torch.pow(1 - target, 4)
    loss = 0
    pred = torch.clamp(pred, 1e-4)

    # Calculate Focal Loss.
    # The hard to classify sample weight is large, easy to classify sample weight is small.
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds * neg_weights 
    
    # Loss normalization is carried out
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        loss = loss - neg_loss
        print("no pos")
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    if torch.isnan(loss):
        print('prediction tensor : ', pred)
        print('prediction pos inds : ', pos_inds)
        print('prediction neg inds : ', neg_inds)


        print('target tensor : ', target)

    return loss


def l1_loss(pred, target, mask):
    """
    Calculate l1 loss
    Args:
        pred: offset detection result
        target: offset ground truth
        mask: offset mask, only center point is 1, other place is 0

    Returns: l1 loss

    """
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    # Don't calculate loss in the position without ground truth.
    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')

    loss = loss / (mask.sum() + 1e-7)

    return loss

def GIoU_loss(boxa, boxb):
    """
    # 为了解决当两个bbox不相交时，距离远的和距离近的损失值一样大。我们认为距离近的损失应该小一点。
    # 注意：划分anchor是否是正样本的时候，anchor与label不一定相交，这样giou能够起到积极的作用
    # 当用正样本计算与label的iou损失时，这时候正样本与label都是相交的情况，这时候GIoU不一定起到积极的作用。
    giou = iou-(|ac-u|)/|ac|   ac最小闭包区域，u并集
    loss = 1 - giou
    """
    inter_x1, inter_y1 = torch.maximum(boxa[:, 0], boxb[:, 0]), torch.maximum(boxa[:, 1], boxb[:, 1])
    inter_x2, inter_y2 = torch.minimum(boxa[:, 2], boxb[:, 2]), torch.minimum(boxa[:, 3], boxb[:, 3])
    inter_h = torch.maximum(torch.tensor([0]), inter_y2 - inter_y1)
    inter_w = torch.maximum(torch.tensor([0]), inter_x2 - inter_x1)
    inter_area = inter_w * inter_h
    union_area = ((boxa[:, 3] - boxa[:, 1]) * (boxa[:, 2] - boxa[:, 0])) + \
                 ((boxb[:, 3] - boxb[:, 1]) * (boxb[:, 2] - boxb[:, 0])) - inter_area + 1e-8  # + 1e-8 防止除零

    # 求最小闭包区域的x1,y1,x2,y2,h,w,area
    ac_x1, ac_y1 = torch.minimum(boxa[:, 0], boxb[:, 0]), torch.minimum(boxa[:, 1], boxb[:, 1])
    ac_x2, ac_y2 = torch.maximum(boxa[:, 2], boxb[:, 2]), torch.maximum(boxa[:, 3], boxb[:, 3])
    ac_w = ac_x2 - ac_x1
    ac_h = ac_y2 - ac_y1
    ac_area = ac_w * ac_h

    giou = (inter_area / union_area) - (torch.abs(ac_area - union_area) / ac_area)
    giou_loss = 1 - giou
    return giou_loss

class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.diou_loss = DIoULoss()
    def forward(self, prediction:dict, groundTrue:dict):
        c_loss = focal_loss(prediction["hms"], groundTrue["hms"])
        wh_loss = 0.1 * l1_loss(prediction["whs"], groundTrue["whs"], groundTrue["masks"])
        off_loss = l1_loss(prediction["offsets"], groundTrue["offsets"], groundTrue["masks"])
        # iou_loss = self.diou_loss(groundTrue["bboxes"], prediction["bboxes"])

        return c_loss + wh_loss + off_loss

