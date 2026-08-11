"""mAP@0.5 and the confusion matrix behind it.

mIOU only scores how well already-matched boxes overlap. mAP is what the competition
grades: a per-class PR curve, so precision, class correctness and score ordering all
count, and every class weighs 1/34 regardless of how many boxes it has.
"""
import numpy as np

# The semantically overlapping groups from DATASET.md 7.3. Classes not listed here are
# visually distinctive and form their own group of one.
GROUPS = {
    "container": ["plastic_bottle", "non_food_plastic_bottle", "glass_bottle", "metal_can",
                  "non_pet_food_beverage_container", "drink_carton", "non_pet_food_container",
                  "non_food_plastic_container", "aluminum_packaging"],
    "cup_tableware": ["takeaway_beverage_cup", "cup", "disposable_food_container",
                      "disposable_tableware", "foam_container", "plastic_lid"],
    "net_rope": ["fishing_net_rope", "net_like_item", "fishing_gear", "fish_trap_and_bait"],
    "float": ["foam_buoy_float", "fishing_buoy_float", "soft_float"],
    "fragment": ["anthropogenic_fragment", "other", "textile"],
}


def group_of(class_names):
    """class index -> group name. Anything unlisted is its own group."""
    lookup = {c: g for g, members in GROUPS.items() for c in members}
    return [lookup.get(n, n) for n in class_names]


def iou_matrix(a, b):
    """a: (N, 4) xyxy, b: (M, 4) xyxy -> (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def average_precision(scores, is_tp, n_gt):
    """All-point interpolated AP. nan when the class has no ground truth at all."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(1 - is_tp[order])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    # Monotonic envelope, then the area under the step function.
    mrec = np.concatenate([[0.0], rec, [rec[-1]]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


class DetectionEval:
    """Accumulate detections image by image, then report mAP@0.5 and where errors go.

    Classes with no ground truth are excluded from the mean rather than scored 0.
    stratified_split() deliberately keeps the 5 rarest classes out of validation, and
    counting them as 0 would silently subtract 15% from every number.
    """

    def __init__(self, num_classes, iou_thres=0.5):
        self.num_classes = num_classes
        self.iou_thres = iou_thres
        self.scores = [[] for _ in range(num_classes)]
        self.tps = [[] for _ in range(num_classes)]
        self.n_gt = np.zeros(num_classes, np.int64)
        # rows = ground truth, cols = what it was called. Last row/col is background.
        self.cm = np.zeros((num_classes + 1, num_classes + 1), np.int64)

    def update(self, preds, gts):
        """preds: (M, 6) x1y1x2y2, score, label.  gts: (N, 5) x1y1x2y2, label."""
        preds = np.asarray(preds, np.float32).reshape(-1, 6)
        gts = np.asarray(gts, np.float32).reshape(-1, 5)
        B = self.num_classes

        for c in range(B):
            p = preds[preds[:, 5] == c]
            g = gts[gts[:, 4] == c]
            self.n_gt[c] += len(g)
            if len(p) == 0:
                continue
            p = p[np.argsort(-p[:, 4])]
            ious = iou_matrix(p[:, :4], g[:, :4])
            taken = np.zeros(len(g), bool)
            tp = np.zeros(len(p), np.float32)
            for i in range(len(p)):
                if len(g) == 0:
                    break
                j = int(np.argmax(np.where(taken, -1, ious[i])))
                if ious[i, j] >= self.iou_thres and not taken[j]:
                    taken[j] = True
                    tp[i] = 1
            self.scores[c].append(p[:, 4])
            self.tps[c].append(tp)

        # Confusion: score each ground truth by its best-overlapping detection of any
        # class. This is the "what did it think this was" view, so it uses one
        # detection per ground truth even when topk emitted several.
        ious = iou_matrix(gts[:, :4], preds[:, :4])
        used = set()
        for i in range(len(gts)):
            cand = np.nonzero(ious[i] >= self.iou_thres)[0]
            cand = [j for j in cand if j not in used]
            t = int(gts[i, 4])
            if not cand:
                self.cm[t, B] += 1                       # missed
                continue
            j = max(cand, key=lambda k: preds[k, 4])
            used.add(j)
            self.cm[t, int(preds[j, 5])] += 1
        for j in range(len(preds)):
            if j not in used and (len(gts) == 0 or ious[:, j].max() < self.iou_thres):
                self.cm[B, int(preds[j, 5])] += 1        # false positive on background

    def per_class_ap(self):
        return np.array([
            average_precision(
                np.concatenate(self.scores[c]) if self.scores[c] else np.zeros(0, np.float32),
                np.concatenate(self.tps[c]) if self.tps[c] else np.zeros(0, np.float32),
                int(self.n_gt[c]))
            for c in range(self.num_classes)])

    def mean_ap(self):
        ap = self.per_class_ap()
        return float(np.nanmean(ap)) if not np.all(np.isnan(ap)) else 0.0

    def report(self, class_names, top_errors=8):
        """Human-readable per-class AP plus where the classification errors land."""
        ap = self.per_class_ap()
        lines = [f"mAP@0.5 = {self.mean_ap():.4f}"
                 f"   ({int((~np.isnan(ap)).sum())}/{self.num_classes} classes have GT)",
                 f"{'class':32}{'GT':>7}{'AP':>9}"]
        for c in np.argsort(np.nan_to_num(ap, nan=-1)):
            v = "  n/a" if np.isnan(ap[c]) else f"{ap[c]:9.3f}"
            lines.append(f"{class_names[c][:31]:32}{self.n_gt[c]:7}{v:>9}")

        groups = group_of(class_names)
        B = self.num_classes
        hit = wrong_in = wrong_out = 0
        for t in range(B):
            for p in range(B):
                n = self.cm[t, p]
                if t == p:
                    hit += n
                elif groups[t] == groups[p]:
                    wrong_in += n
                else:
                    wrong_out += n
        missed = int(self.cm[:B, B].sum())
        fp = int(self.cm[B, :B].sum())
        total = hit + wrong_in + wrong_out + missed
        if total:
            lines += ["",
                      f"localised and named right   {hit:7} ({hit/total:6.1%})",
                      f"localised, WRONG same group {wrong_in:7} ({wrong_in/total:6.1%})",
                      f"localised, wrong other group{wrong_out:7} ({wrong_out/total:6.1%})",
                      f"not localised at all        {missed:7} ({missed/total:6.1%})",
                      f"false positives on background {fp}"]
            if wrong_in + wrong_out:
                lines.append(f"-> {wrong_in/(wrong_in+wrong_out):.0%} of naming errors stay "
                             f"inside the semantic group")

        pairs = [(self.cm[t, p], class_names[t], class_names[p])
                 for t in range(B) for p in range(B) if t != p and self.cm[t, p]]
        if pairs:
            lines.append("\ntop confusions (ground truth -> called)")
            for n, t, p in sorted(pairs, reverse=True)[:top_errors]:
                same = "same group" if group_of(class_names)[class_names.index(t)] == \
                                       group_of(class_names)[class_names.index(p)] else ""
                lines.append(f"  {n:5}  {t[:28]:30} -> {p[:28]:30} {same}")
        return "\n".join(lines)


def proposal_recall(records, iou_threshold=0.5, topk=50):
    """Fraction of ground-truth boxes covered by any of the top-K proposals."""
    hit = total = 0
    for record in records:
        gt = np.asarray(record["gt_boxes"], np.float32).reshape(-1, 4)
        proposals = np.asarray(record["proposals"], np.float32).reshape(-1, 5)[:topk, :4]
        total += len(gt)
        if len(gt) and len(proposals):
            hit += int((iou_matrix(gt, proposals).max(1) >= iou_threshold).sum())
    return hit / max(1, total)
