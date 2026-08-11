import os
import cv2
import json
import random
import torch
import pickle
import numpy as np
from utils.detect import postprocess_output, decode_bbox, roi_rerank
from pt_dataset.dataUtils import to_chw_tensor, read_image_rgb

def folderCheck(folders:list):
    for path in folders:
        os.makedirs(path, exist_ok=True)


def load_annotation(file_path, image_root=None):
    '''
    Load a COCO style json (images / categories / annotations).

    file_path  : path to the json, e.g. data/train_dataset/train_label.json
    image_root : folder holding the image files, defaults to <json folder>/images

    return (annotations, class_names) where annotations is one entry per image:
    [
        {
          "path": local image path,
          "bbox": [[x1, y1, x2, y2], ...],
          "labels": [category_id, ...]
        },
        ...
    ]
    '''
    with open(file_path, 'r') as file:
        data = json.load(file)

    if image_root is None:
        image_root = os.path.join(os.path.dirname(file_path), "images")

    images = {im["id"]: im for im in data["images"]}
    grouped = {}
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            # A zero area box has no center to encode and produces nan loss.
            continue
        item = grouped.setdefault(a["image_id"], {"bbox": [], "labels": []})
        # COCO stores [x, y, w, h], everything downstream is [x1, y1, x2, y2].
        item["bbox"].append([x, y, x + w, y + h])
        item["labels"].append(a["category_id"])

    annotations = [{"path": os.path.join(image_root, images[img_id]["filename"]),
                    "bbox": item["bbox"],
                    "labels": item["labels"]}
                   for img_id, item in grouped.items()]

    class_names = [c["name"] for c in sorted(data["categories"], key=lambda c: c["id"])]

    return annotations, class_names


def stratified_split(annotations, save_path="data/", valid_ratio=0.2, seed=42, save=True,
                     rare_threshold=150):
    '''
    Split into train/valid so that every category keeps roughly valid_ratio of its
    images in the validation set.

    An image can carry several categories, so each image is assigned once and
    categories are filled rarest first: a rare category has almost no images to
    choose from, a common one can always reach its quota from whatever is left.

    rare_threshold: categories with fewer boxes than this keep every one of their
    images in train. 20% of aluminum_packaging is 3 boxes, which is too few to
    estimate an AP from and is 23% of the only training signal that class has.
    The class is still learned and still scored on the real test set, it just has
    no local validation number. Set to 0 to validate on everything.

    Fixed seed: reshuffling between runs leaks validation images into training
    and makes "best" checkpoints from different runs incomparable.
    '''
    by_cat = {}
    box_count = {}
    for i, item in enumerate(annotations):
        for cat in item["labels"]:
            box_count[cat] = box_count.get(cat, 0) + 1
        for cat in set(item["labels"]):
            by_cat.setdefault(cat, []).append(i)

    rare = {cat for cat, n in box_count.items() if n < rare_threshold}
    # An image holding a rare class is pinned to train even if it also holds common ones.
    train_only = {i for cat in rare for i in by_cat[cat]}

    rng = random.Random(seed)
    valid_idx = set()
    for cat in sorted(by_cat, key=lambda c: len(by_cat[c])):
        if cat in rare:
            continue
        members = by_cat[cat]
        # Quota is a share of the whole category, drawn from whatever is not pinned.
        quota = round(len(members) * valid_ratio)
        already = sum(1 for i in members if i in valid_idx)
        pool = [i for i in members if i not in valid_idx and i not in train_only]
        rng.shuffle(pool)
        valid_idx.update(pool[:max(0, quota - already)])

    train_annotation = [a for i, a in enumerate(annotations) if i not in valid_idx]
    valid_annotation = [a for i, a in enumerate(annotations) if i in valid_idx]

    # save=False for non-zero DDP ranks: every rank computes the same split from the
    # same seed, and concurrent writes to one file would interleave.
    if save:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'train.pkl'), 'wb') as file:
            pickle.dump(train_annotation, file)

        with open(os.path.join(save_path, 'valid.pkl'), 'wb') as file:
            pickle.dump(valid_annotation, file)

    return train_annotation, valid_annotation


def read_imgTotensor(path, image_size):
    '''Returns (RGB uint8 image for drawing, normalised NCHW tensor for the model).'''
    image = cv2.resize(read_image_rgb(path), image_size)

    # Same preprocessing as ImgDataset, otherwise inference sees a different distribution.
    input_data = torch.from_numpy(to_chw_tensor(image)).unsqueeze(0)

    return image, input_data

def predict_full(model, input_data, image_size, conf, nms_thres, dev, roi_topk=3):
    """Both stages: detect, then re-score each box with the RoI classifier.

    Needs the real CenterNet module, not the jit-traced one: tracing follows the
    rois=None branch and so has no second stage.
    """
    x = input_data.to(dev)
    hms, whs, offsets = model(x)
    dets = decoder(hms, whs, offsets, image_size, conf, nms_thres, dev)
    if getattr(model, "roi_head", None) is None or len(dets) == 0:
        return dets
    logits = model(x, [dets[:, :4].float()])[3]
    return roi_rerank(dets, logits, roi_topk)


def decoder(hms, whs, offsets, image_size, conf, nms_thres, dev):
    """
    Turn raw model outputs (NCHW) into boxes for one image.
    Args:
        hms, whs, offsets: model outputs, shape (N, C, H, W)
        image_size: (w, h) of the model input
        conf: heatmap confidence threshold
        nms_thres: nms threshold
        dev: torch device

    Returns:  bounding box of one image(x1, y1, x2, y2, score, label).

    """
    hms = hms.permute(0, 2, 3, 1)
    whs = whs.permute(0, 2, 3, 1)
    offsets = offsets.permute(0, 2, 3, 1)

    outputs = postprocess_output(hms, whs, offsets, conf, dev)
    outputs = decode_bbox(outputs,
                          image_size,
                          dev, image_shape=image_size, remove_pad=True,
                          need_nms=True, nms_thres=nms_thres)

    return outputs[0]

def predict(input_data, image_size, conf, nms_thres, model, dev):
    """Run the model on one preprocessed image and decode it. Returns (x1, y1, x2, y2, score, label)."""
    hms, whs, offsets = model(input_data.to(dev))
    return decoder(hms, whs, offsets, image_size, conf, nms_thres, dev)

def draw_bbox(image, bboxes, labels, class_names, color_map, scores=None, show_name=False):
    """
    Draw bounding box in image.
    Args:
        image: image
        bboxes: coordinate of bounding box
        labels: the index of labels
        class_names: the names of class
        scores: bounding box confidence
        show_name: show class name if set true, otherwise show index of class

    Returns: draw result

    """
    
    image_height, image_width = image.shape[:2]
    draw_image = image.copy()

    for i, c in list(enumerate(labels)):
        bbox = bboxes[i]
        c = int(c)
        color = [int(j) for j in color_map[c]]
        if show_name:
            predicted_class = class_names[c]
        else:
            predicted_class = c

        if scores is None:
            text = '{}'.format(predicted_class)
        else:
            score = scores[i]
            text = '{} {:.2f}'.format(predicted_class, score)

        x1, y1, x2, y2 = bbox
        x1 = max(0, np.floor(x1).astype(np.int32))
        y1 = max(0, np.floor(y1).astype(np.int32))
        x2 = min(image_width, np.floor(x2).astype(np.int32))
        y2 = min(image_height, np.floor(y2).astype(np.int32))

        thickness = int((image_height + image_width) / (np.sqrt(image_height**2 + image_width**2)))
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), color=color, thickness=thickness)
        cv2.putText(draw_image, text, (x1, max(0, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return draw_image


def collapse_annotations(annotations):
    """Copy annotations while replacing every foreground label with object class 0."""
    return [{**item, "labels": [0] * len(item["labels"])} for item in annotations]


def load_stage2_detector(weights_path, backbone, num_classes, device, class_agnostic=False):
    """Load a detector checkpoint without downloading redundant backbone weights."""
    from model.backnone import build_backbone
    from model.centerNet import CenterNet
    from utils.pytorchtools import load_weights

    backbone_module = build_backbone(backbone, weights=None)
    model = CenterNet(
        num_classes=1 if class_agnostic else num_classes,
        backbone=backbone_module,
        roi_head=not class_agnostic,
    ).to(device)
    blob = torch.load(weights_path, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    load_weights(model, state, str(weights_path))
    model.eval()
    return model


def _scale_to_original(detections, raw_w, raw_h, input_shape):
    if len(detections) == 0:
        return detections
    detections = detections.clone()
    detections[:, [0, 2]] *= raw_w / input_shape[0]
    detections[:, [1, 3]] *= raw_h / input_shape[1]
    detections[:, [0, 2]].clamp_(0, raw_w)
    detections[:, [1, 3]].clamp_(0, raw_h)
    return detections


@torch.no_grad()
def infer_stage2_proposals(model, image, device, input_shape=(512, 512), confidence=0.001,
                           topk=1000, proposal_nms=0.7, max_proposals=100,
                           keep_stage1=False):
    """Return class-agnostic proposals in original-image pixels."""
    from utils.detect import class_agnostic_nms
    from utils.pytorchtools import amp_dtype

    raw_h, raw_w = image.shape[:2]
    resized = cv2.resize(image, input_shape)
    x = torch.from_numpy(to_chw_tensor(resized)).unsqueeze(0).to(device)
    with torch.autocast(device.type, dtype=amp_dtype(device), enabled=device.type != "cpu"):
        hms, whs, offsets = model(x)[:3]
    hms, whs, offsets = [v.permute(0, 2, 3, 1).float() for v in (hms, whs, offsets)]
    raw = postprocess_output(hms, whs, offsets, confidence, device, topk=topk)[0]
    if len(raw) == 0:
        return np.zeros((0, 5), np.float32), np.zeros((0, 6), np.float32)

    decoded = decode_bbox([raw.clone()], input_shape, device, need_nms=False)[0]
    proposals = class_agnostic_nms(decoded, proposal_nms, max_proposals)
    proposals = _scale_to_original(proposals, raw_w, raw_h, input_shape)
    stage1 = torch.zeros((0, 6), device=device)
    if keep_stage1:
        stage1 = decode_bbox(
            [raw.clone()], input_shape, device, need_nms=True, nms_thres=0.45)[0]
        stage1 = _scale_to_original(stage1, raw_w, raw_h, input_shape)
    return proposals.cpu().numpy(), stage1.cpu().numpy()


def generate_proposal_records(annotations, model, device, *, keep_stage1=False,
                              input_shape=(512, 512), confidence=0.001, topk=1000,
                              proposal_nms=0.7, max_proposals=100):
    from tqdm import tqdm

    records = []
    for item in tqdm(annotations, desc="proposals", ascii=" ="):
        image = read_image_rgb(item["path"])
        proposals, stage1 = infer_stage2_proposals(
            model, image, device, input_shape, confidence, topk,
            proposal_nms, max_proposals, keep_stage1)
        records.append({
            "path": item["path"],
            "gt_boxes": [[float(v) for v in box] for box in item["bbox"]],
            "gt_labels": [int(v) for v in item["labels"]],
            "proposals": proposals.tolist(),
            "detections": stage1.tolist() if keep_stage1 else [],
        })
    return records


def proposal_cache_metadata(weights_path, backbone, class_agnostic, *, keep_stage1=False,
                            input_shape=(512, 512), confidence=0.001, topk=1000,
                            proposal_nms=0.7, max_proposals=100):
    from pathlib import Path

    path = Path(weights_path)
    stat = path.stat()
    return {
        "weights": str(path.resolve()),
        "weights_size": stat.st_size,
        "weights_mtime_ns": stat.st_mtime_ns,
        "backbone": backbone,
        "class_agnostic": class_agnostic,
        "keep_stage1": keep_stage1,
        "input_shape": list(input_shape),
        "confidence": confidence,
        "topk": topk,
        "proposal_nms": proposal_nms,
        "max_proposals": max_proposals,
    }


def load_or_generate_proposal_records(path, annotations, model, device, metadata,
                                      rebuild=False):
    from pathlib import Path

    path = Path(path)
    if path.exists() and not rebuild:
        with path.open() as file:
            payload = json.load(file)
        if payload.get("metadata") != metadata:
            raise RuntimeError(f"stale proposal cache: {path}; use --rebuild-manifest")
        return payload["records"]
    records = generate_proposal_records(
        annotations, model, device,
        keep_stage1=metadata["keep_stage1"],
        input_shape=tuple(metadata["input_shape"]),
        confidence=metadata["confidence"], topk=metadata["topk"],
        proposal_nms=metadata["proposal_nms"],
        max_proposals=metadata["max_proposals"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as file:
        json.dump({"metadata": metadata, "records": records}, file)
    os.replace(tmp, path)
    return records


def gt_crop_samples(records):
    return [{"path": record["path"], "box": box, "label": int(label)}
            for record in records
            for box, label in zip(record["gt_boxes"], record["gt_labels"])]


def crop_samples_from_records(records, positive_iou=0.5, negative_iou=0.3,
                              positives_per_gt=2, negatives_per_gt=3,
                              background=34):
    """Build GT, detector-noise positive, and hard-background classifier samples."""
    from utils.metrics import iou_matrix

    samples = gt_crop_samples(records)
    for record in records:
        proposals = np.asarray(record["proposals"], np.float32).reshape(-1, 5)
        gt = np.asarray(record["gt_boxes"], np.float32).reshape(-1, 4)
        labels = np.asarray(record["gt_labels"], np.int64)
        if len(proposals) == 0:
            continue
        if len(gt):
            ious = iou_matrix(proposals[:, :4], gt)
            best_gt = ious.argmax(1)
            best_iou = ious[np.arange(len(proposals)), best_gt]
        else:
            best_gt = np.zeros(len(proposals), np.int64)
            best_iou = np.zeros(len(proposals), np.float32)

        for gt_index in range(len(gt)):
            candidates = np.where((best_gt == gt_index) & (best_iou >= positive_iou))[0]
            candidates = candidates[np.argsort(-proposals[candidates, 4])]
            for proposal_index in candidates[:positives_per_gt]:
                samples.append({
                    "path": record["path"],
                    "box": proposals[proposal_index, :4].tolist(),
                    "label": int(labels[gt_index]),
                })
        negatives = np.where(best_iou < negative_iou)[0]
        negatives = negatives[np.argsort(-proposals[negatives, 4])]
        for proposal_index in negatives[:max(1, len(gt)) * negatives_per_gt]:
            samples.append({
                "path": record["path"],
                "box": proposals[proposal_index, :4].tolist(),
                "label": background,
            })
    return samples


def crop_sample_weights(samples, background=34, background_fraction=0.25, cap=8.0):
    import math
    from collections import Counter

    counts = Counter(int(sample["label"]) for sample in samples)
    foreground = {label: count for label, count in counts.items() if label != background}
    if not foreground:
        raise ValueError("crop training set has no foreground samples")
    largest = max(foreground.values())
    per_class = {label: min(cap, math.sqrt(largest / count))
                 for label, count in foreground.items()}
    weights = np.array([per_class.get(int(sample["label"]), 0.0) for sample in samples])
    background_mask = np.array([int(sample["label"]) == background for sample in samples])
    if background_mask.any():
        foreground_mass = weights[~background_mask].sum()
        weights[background_mask] = (
            background_fraction / (1 - background_fraction)
            * foreground_mass / background_mask.sum())
    return torch.as_tensor(weights, dtype=torch.double)


@torch.no_grad()
def classify_proposal_records(model, records, device, batch_size=128, workers=8,
                              size=224, expand=0.1):
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from pt_dataset.dataset import ProposalCropDataset
    from utils.pytorchtools import amp_dtype

    classes = model.net.classifier[-1].out_features
    outputs = [np.zeros((len(record["proposals"]), classes), np.float32)
               for record in records]
    dataset = ProposalCropDataset(records, size=size, expand=expand)
    if not len(dataset):
        return outputs
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        persistent_workers=workers > 0, pin_memory=device.type == "cuda")
    model.eval()
    for images, record_ids, proposal_ids in tqdm(loader, desc="classify", ascii=" ="):
        images = images.to(device)
        with torch.autocast(device.type, dtype=amp_dtype(device), enabled=device.type != "cpu"):
            probabilities = model(images).softmax(-1).float().cpu().numpy()
        for probabilities_i, record_id, proposal_id in zip(
                probabilities, record_ids.tolist(), proposal_ids.tolist()):
            outputs[record_id][proposal_id] = probabilities_i
    return outputs


def _record_ground_truth(record):
    boxes = np.asarray(record["gt_boxes"], np.float32).reshape(-1, 4)
    labels = np.asarray(record["gt_labels"], np.float32).reshape(-1, 1)
    return np.concatenate([boxes, labels], axis=1)


def evaluate_stage2_records(model, records, device, num_classes=34, batch_size=128,
                            workers=8, size=224, expand=0.1, score_threshold=0.001,
                            nms_threshold=0.45, max_detections=100):
    from utils.detect import classifier_detections
    from utils.metrics import DetectionEval

    probabilities = classify_proposal_records(
        model, records, device, batch_size, workers, size, expand)
    evaluator = DetectionEval(num_classes)
    detections = []
    for record, record_probabilities in zip(records, probabilities):
        dets = classifier_detections(
            record["proposals"], record_probabilities, num_classes,
            score_threshold, nms_threshold, max_detections).numpy()
        evaluator.update(dets, _record_ground_truth(record))
        detections.append(dets)
    return evaluator.mean_ap(), evaluator, detections


def evaluate_stage1_records(records, num_classes=34):
    from utils.metrics import DetectionEval

    evaluator = DetectionEval(num_classes)
    for record in records:
        evaluator.update(
            np.asarray(record.get("detections", []), np.float32).reshape(-1, 6),
            _record_ground_truth(record))
    return evaluator.mean_ap(), evaluator


def select_proposal_topk(records, candidates=(10, 20, 50, 100), retain=0.99):
    from utils.metrics import proposal_recall

    baseline = proposal_recall(records, 0.5, max(candidates))
    if baseline == 0:
        return max(candidates)
    for topk in candidates:
        if proposal_recall(records, 0.5, topk) >= baseline * retain:
            return topk
    return max(candidates)


def limit_record_proposals(records, topk):
    return [{**record, "proposals": record["proposals"][:topk]} for record in records]


@torch.no_grad()
def evaluate_crop_classifier(model, samples, device, batch_size=128, workers=8,
                             size=224, expand=0.1):
    import torch.nn.functional as F
    from collections import Counter
    from torch.utils.data import DataLoader
    from pt_dataset.dataset import CropDataset
    from utils.pytorchtools import amp_dtype

    loader = DataLoader(
        CropDataset(samples, train=False, size=size, expand=expand),
        batch_size=batch_size, shuffle=False, num_workers=workers,
        persistent_workers=workers > 0, pin_memory=device.type == "cuda")
    loss_sum = correct = total = 0
    per_class_total, per_class_correct = Counter(), Counter()
    model.eval()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device.type, dtype=amp_dtype(device), enabled=device.type != "cpu"):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        prediction = logits.argmax(1)
        loss_sum += loss.item() * len(labels)
        correct += (prediction == labels).sum().item()
        total += len(labels)
        for label, predicted in zip(labels.cpu().tolist(), prediction.cpu().tolist()):
            per_class_total[label] += 1
            per_class_correct[label] += int(label == predicted)
    recalls = [per_class_correct[label] / count
               for label, count in per_class_total.items()]
    return {
        "loss": loss_sum / max(1, total),
        "accuracy": correct / max(1, total),
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
    }
