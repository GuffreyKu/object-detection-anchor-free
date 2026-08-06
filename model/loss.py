import torch
import torch.nn as nn
import torch.nn.functional as F

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
    # Clamp both ends: sigmoid rounds to exactly 1.0 in fp32 and log(1 - 1.0) is -inf.
    pred = torch.clamp(pred, 1e-4, 1 - 1e-4)

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
        # Dumping the tensors here buried every other line of output. The counts say
        # as much: nan reaches this point only via a nan in pred, since clamp() passes
        # nan through untouched.
        print(f"focal_loss nan: {int(torch.isnan(pred).sum())} nan / "
              f"{int(torch.isinf(pred).sum())} inf in prediction, num_pos={int(num_pos)}")

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

class TotalLoss(nn.Module):
    def __init__(self, roi_weight=1.0):
        super().__init__()
        self.roi_weight = roi_weight

    def forward(self, prediction:dict, groundTrue:dict):
        c_loss = focal_loss(prediction["hms"], groundTrue["hms"])
        wh_loss = 0.1 * l1_loss(prediction["whs"], groundTrue["whs"], groundTrue["masks"])
        off_loss = l1_loss(prediction["offsets"], groundTrue["offsets"], groundTrue["masks"])
        loss = c_loss + wh_loss + off_loss

        # Second stage. Plain cross entropy, so the classes inside one semantic group
        # compete for the same probability mass - which the 34 independent sigmoids in
        # the heatmap never make them do (DATASET.md 12.3).
        logits = prediction.get("roi_logits")
        if logits is not None and logits.shape[0] > 0:
            loss = loss + self.roi_weight * F.cross_entropy(logits, groundTrue["roi_labels"])

        return loss

