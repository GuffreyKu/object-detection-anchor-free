import math

import numpy as np
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

def inverse_sqrt_class_weights(counts, clip=(0.25, 4.0)):
    """
    w_c proportional to 1/sqrt(n_c), normalised to mean 1 and clipped.

    mAP averages AP with every class weighing 1/34 while the boxes are imbalanced 470:1,
    so an unweighted mean-reduction cross entropy optimises a very different objective
    from the one being scored. Inverse-sqrt rather than inverse-frequency, and clipped,
    because the full 470x correction would put most of the gradient on a 13-box class.
    """
    counts = np.asarray(counts, np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    w = w / w.mean()
    return torch.tensor(np.clip(w, *clip), dtype=torch.float32)


class TotalLoss(nn.Module):
    def __init__(self, roi_weight=1.0, class_weights=None):
        """
        class_weights: optional (num_classes,) tensor for the second-stage cross entropy.
                       Registered as a buffer so .to(DEVICE) moves it and it survives a
                       state_dict round trip.
        """
        super().__init__()
        self.roi_weight = roi_weight
        self.register_buffer("class_weights",
                             None if class_weights is None else class_weights.float())

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
            loss = loss + self.roi_weight * F.cross_entropy(
                logits, groundTrue["roi_labels"], weight=self.class_weights)

        return loss


class HardNegativeCrossEntropy(nn.Module):
    """Cross entropy plus a margin over the hardest class in the same confusion group.

    arc_margin/arc_scale optionally add an ArcFace-style additive angular margin (Deng et
    al. 2019) ahead of the softmax. When enabled, `logits` must be COSINE SIMILARITIES in
    [-1, 1] - i.e. the classifier's final layer has to be a normalised-embedding /
    normalised-weight head (CropClassifier(arc_margin=True) in model/centerNet.py), not a
    plain Linear. The margin is applied here rather than in the model because it only
    touches the true class's angle and needs the label, which forward(x) alone doesn't
    have. The hard-negative margin term below still reads the pre-margin, pre-scale
    cosine values, so `margin` keeps the same [-1, 1]-scale meaning whether or not arc
    mode is on.

    arc_margin=0 (the default) reproduces the exact prior behaviour bit-for-bit. This is
    experimental and untested against this dataset's class imbalance (13 to 6000+
    samples/class): an angular margin asks every class to carve out its own angular
    region, and the rarest classes may not have enough samples to do that safely. See
    docs/STAGE2_EVAL.md.
    """

    def __init__(self, num_classes, class_weights=None, hard_negative_groups=(),
                 hard_weight=0.5, margin=0.2, label_smoothing=0.05,
                 arc_margin=0.0, arc_scale=30.0):
        super().__init__()
        if hard_weight < 0 or margin < 0:
            raise ValueError("hard-negative weight and margin must be non-negative")
        if arc_margin < 0 or arc_scale <= 0:
            raise ValueError("arc margin must be non-negative and arc scale positive")
        self.hard_weight = hard_weight
        self.margin = margin
        self.label_smoothing = label_smoothing
        self.arc_margin = arc_margin
        self.arc_scale = arc_scale
        self.register_buffer(
            "class_weights", None if class_weights is None else class_weights.float())

        mask = torch.zeros((num_classes, num_classes), dtype=torch.bool)
        for group in hard_negative_groups:
            group = torch.as_tensor(group, dtype=torch.long)
            if len(group) and (group.min() < 0 or group.max() >= num_classes):
                raise ValueError("hard-negative group contains an invalid class index")
            for label in group:
                mask[label, group] = True
        mask.fill_diagonal_(False)
        self.register_buffer("hard_negative_mask", mask)

    def _arc_logits(self, cosine, targets):
        """cosine: (N, C) in [-1, 1]. Adds `arc_margin` to the true class's angle only,
        then rescales - the standard ArcFace transform."""
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_theta = theta.gather(1, targets[:, None]) + self.arc_margin
        target_cosine = torch.cos(target_theta.clamp(max=math.pi))
        return cosine.scatter(1, targets[:, None], target_cosine) * self.arc_scale

    def forward(self, logits, targets):
        if logits.shape[-1] != self.hard_negative_mask.shape[0]:
            raise ValueError("classifier output does not match hard-negative classes")
        raw = logits
        if self.arc_margin > 0:
            logits = self._arc_logits(logits, targets)
        loss = F.cross_entropy(
            logits, targets, weight=self.class_weights,
            label_smoothing=self.label_smoothing)
        eligible = self.hard_negative_mask[targets]
        valid = eligible.any(dim=1)
        if self.hard_weight == 0 or not valid.any():
            return loss

        hard_logits = raw[valid].masked_fill(~eligible[valid], -torch.inf).max(1).values
        true_logits = raw[valid].gather(1, targets[valid, None]).squeeze(1)
        hard_loss = F.relu(hard_logits + self.margin - true_logits)
        if self.class_weights is not None:
            weights = self.class_weights[targets[valid]]
            hard_loss = (hard_loss * weights).sum() / weights.sum().clamp_min(1e-12)
        else:
            hard_loss = hard_loss.mean()
        return loss + self.hard_weight * hard_loss
