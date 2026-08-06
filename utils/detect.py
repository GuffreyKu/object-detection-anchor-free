import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

def peak_filter(hms, kernel=3):
    """
    Keep only local maxima of the heatmap.
    One object lights up a whole gaussian blob, but wh/offset are only written at its
    center. Without this the blob's neighbours decode into zero-area boxes that NMS
    cannot suppress. Args: hms (n, h, w, c). Returns: hms with non-peaks zeroed.
    """
    hms = hms.permute(0, 3, 1, 2)
    keep = (F.max_pool2d(hms, kernel_size=kernel, stride=1, padding=(kernel - 1) // 2) == hms)
    return (hms * keep.float()).permute(0, 2, 3, 1)


def postprocess_output(hms, whs, offsets, confidence, dev, topk=100):
    """
    The post process of model output.
    Args:
        hms: heatmap, (n, h, w, c)
        whs: the height and width of bounding box
        offsets: center point offset
        confidence: the threshold of heatmap
        dev: torch device
        topk: how many (cell, class) peaks to keep per image

    Returns:  The list of bounding box(x, y, w, h, score, label).

    Peaks are taken over the whole (h*w*c) volume, not argmax over c. A cell may
    therefore emit several classes. AP scores one PR curve per class and needs a
    ranked candidate for every class a box might be; argmax throws away every
    runner-up, which is exactly the confusable-class case (DATASET.md 9.3).
    """
    hms = peak_filter(hms)
    batch, output_h, output_w, c = hms.shape

    # Feature point coordinates are the same for every image in the batch, build them once.
    yv, xv = torch.meshgrid(torch.arange(0, output_h, device=dev),
                            torch.arange(0, output_w, device=dev),
                            indexing='ij')
    xv, yv = xv.flatten().float(), yv.flatten().float()

    detections = []
    for b in range(batch):
        # (h, w, c) -> (hw, c), (h, w, 2) -> (hw, 2)
        heat_map = hms[b].reshape(-1, c)
        wh = whs[b].reshape(-1, 2)
        offset = offsets[b].reshape(-1, 2)

        flat = heat_map.reshape(-1)
        score, idx = flat.topk(min(topk, flat.numel()))
        keep = score > confidence
        score, idx = score[keep], idx[keep]

        if score.numel() == 0:
            detections.append([])
            continue

        # Row-major flatten of (hw, c): cell = idx // c, class = idx % c.
        cell, label = torch.div(idx, c, rounding_mode='floor'), idx % c
        wh_mask, offset_mask = wh[cell], offset[cell]

        # Adjust center of predict box
        xv_mask = torch.unsqueeze(xv[cell] + offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[cell] + offset_mask[..., 1], -1)

        # Get the (xmin, ymin, xmax, ymax)
        half_w, half_h = wh_mask[..., 0:1] / 2, wh_mask[..., 1:2] / 2
        bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)

        # Bounding box coordinate normalize
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h

        # Concatenate the prediction
        detect = torch.cat(
            [bboxes, torch.unsqueeze(score, -1), torch.unsqueeze(label, -1).float()], dim=-1)
        detections.append(detect)

    return detections


def roi_rerank(dets, logits, topk=3):
    """
    Turn one box into several ranked class hypotheses using the RoI head's softmax.
    Args:
        dets: (M, 6) x1y1x2y2, score, label from decode_bbox
        logits: (M, C) RoI classifier output for the same boxes
        topk: class hypotheses to emit per box

    Returns: (M*topk, 6) in the same format.

    Same reason as the topk decode: AP wants a scored candidate per class, not one
    hard decision. Score is the stage-1 objectness times the stage-2 probability.
    """
    if len(dets) == 0:
        return dets
    p = logits.softmax(-1)
    k = min(topk, p.shape[1])
    cls_p, cls_i = p.topk(k, dim=-1)
    box = dets[:, :4].unsqueeze(1).expand(-1, k, -1)
    score = (dets[:, 4:5] * cls_p).unsqueeze(-1)
    return torch.cat([box, score, cls_i.unsqueeze(-1).float()], -1).reshape(-1, 6)


def decode_bbox(prediction, input_shape, dev, image_shape=None, remove_pad=False, need_nms=False, nms_thres=0.4):
    """
    Decode postprecess_output output
    Args:
        prediction: postprecess_output output
        input_shape: model input shape, (w, h)
        dev: torch device
        image_shape: original image shape, (w, h)
        remove_pad: model input is padding image, you should set remove_pad=True if you want to remove this pad
        need_nms: whether use NMS to remove redundant detect box
        nms_thres: nms threshold

    Returns:  The list of bounding box(x1, y1, x2, y2 score, label).

    """
    output = [[] for _ in prediction]
    scale_xyxy = torch.tensor(tuple(input_shape) * 2, device=dev, dtype=torch.float32)

    for b, detection in enumerate(prediction):
        if len(detection) == 0:
            continue

        if need_nms:
            # Per class: boxes of different classes must not suppress each other.
            keep = batched_nms(detection[:, :4], detection[:, 4], detection[:, 5], nms_thres)
            detection = detection[keep]

        output[b] = detection
        bboxes = detection[:, 0:4] * scale_xyxy

        if remove_pad:
            assert image_shape is not None, \
                "If remove_pad is True, image_shape must be set the shape of original image."
            iw, ih = input_shape
            w, h = image_shape
            scale = min(iw/w, ih/h)
            nw, nh = int(scale * w), int(scale * h)
            dw, dh = (iw - nw) // 2, (ih - nh) // 2

            bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - dw) / scale
            bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - dh) / scale

        output[b][:, :4] = bboxes

    return output


