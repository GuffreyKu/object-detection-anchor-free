import numpy as np
import torch
from torch import nn
from torchvision.ops import nms

def postprocess_output(hms, whs, offsets, confidence, dev):
    """
    The post process of model output.
    Args:
        hms: heatmap
        whs: the height and width of bounding box
        offsets: center point offset
        confidence: the threshold of heatmap
        dev: torch device

    Returns:  The list of bounding box(x, y, w, h, score, label).

    """
    batch, output_h, output_w, c = hms.shape

    detections = []
    for b in range(batch):
        # (h, w, c) -> (-1, c)
        heat_map = hms[b].view([-1, c])
        # print("heat_map", heat_map)
        # (h, w, 2) -> (-1, 2)
        wh = whs[b].view([-1, 2])
        # print("wh", wh)

        # (h, w, 2) -> (-1, 2)
        offset = offsets[b].view([-1, 2])
        # print("offset", offset)

        yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))

        xv, yv = xv.flatten().float(), yv.flatten().float()

        xv = xv.to(dev)     # x axis coordinate of feature point
        yv = yv.to(dev)     # y axis coordinate of feature point

        # torch.max[0] max value
        # torch.max[1] index of max value
        score, label = torch.max(heat_map, dim=-1)

        mask = score > confidence

        # Choose height, width and offset by confidence mask
        wh_mask = wh[mask]
        # print("wh_mask", wh_mask)

        offset_mask = offset[mask]

        if len(wh_mask) == 0:
            detections.append([])
            continue

        # Adjust center of predict box
        xv_mask = torch.unsqueeze(xv[mask] + offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + offset_mask[..., 1], -1)

        # print("xv_mask ", xv_mask)
        # print("yv_mask ", yv_mask)

        # Get the (xmin, ymin, xmax, ymax)
        half_w, half_h = wh_mask[..., 0:1] / 2, wh_mask[..., 1:2] / 2
        bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)

        # Bounding box coordinate normalize
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h

        # Concatenate the prediction
        detect = torch.cat(
            [bboxes, torch.unsqueeze(score[mask], -1), torch.unsqueeze(label[mask], -1).float()], dim=-1)
        detections.append(detect)

    return detections


def decode_bbox(prediction, input_shape, dev, image_shape=None, remove_pad=False, need_nms=False, nms_thres=0.4):
    """
    Decode postprecess_output output
    Args:
        prediction: postprecess_output output
        input_shape: model input shape
        dev: torch device
        image_shape: image shape
        remove_pad: model input is padding image, you should set remove_pad=True if you want to remove this pad
        need_nms: whether use NMS to remove redundant detect box
        nms_thres: nms threshold

    Returns:  The list of bounding box(x1, y1, x2, y2 score, label).

    """
    output = [[] for _ in prediction]

    for b, detection in enumerate(prediction):
        if len(detection) == 0:
            continue

        if need_nms:
            keep = nms(detection[:, :4], detection[:, 4], nms_thres)
            detection = detection[keep]

        output[b].append(detection)

        output[b] = torch.cat(output[b])
        if output[b] is not None:
            bboxes = output[b][:, 0:4]

            input_shape = torch.tensor(input_shape, device=dev)
            bboxes *= torch.cat([input_shape, input_shape], dim=-1)

            if remove_pad:
                assert image_shape is not None, \
                    "If remove_pad is True, image_shape must be set the shape of original image."
                ih, iw = input_shape
                h,  w = image_shape
                scale = min(iw/w, ih/h)
                nw, nh = int(scale * w), int(scale * h)
                dw, dh = (iw - nw) // 2, (ih - nh) // 2

                bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - dw) / scale
                bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - dh) / scale

            output[b][:, :4] = bboxes

    return output


