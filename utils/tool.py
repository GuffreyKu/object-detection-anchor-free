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