import torch
import numpy as np
from tqdm import tqdm
from utils.detect import postprocess_output, decode_bbox, roi_rerank
from utils.metrics import DetectionEval

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(now_ep,
          model,
          optimizer,
          scheduler,
          dataloader,
          criterion,
          DEVICE,
          scaler=None):

    losses = []
    amp = scaler is not None and scaler.is_enabled()
    model.train()
    with tqdm(dataloader, ascii=' =', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as loader:
        for image, heat_map, wh, offset, offset_mask, boxes, labels in loader:
            loader.set_description(f"train {now_ep}")
            image = image.to(DEVICE)

            heat_map = heat_map.to(DEVICE)
            wh = wh.to(DEVICE)
            offset = offset.to(DEVICE)
            offset_mask = offset_mask.to(DEVICE)

            # The RoI head is taught on ground-truth boxes. Feeding it the detector's
            # own boxes this early would just have it learn from noise.
            rois = [b.to(DEVICE) for b in boxes]
            roi_labels = torch.cat(labels).to(DEVICE) if len(labels) else None

            optimizer.zero_grad()

            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=amp):
                out = model(image, rois)

            # Back to fp32 for the loss: focal loss takes log() of a sigmoid and
            # fp16 underflows there.
            hms_pred = out[0].permute(0, 2, 3, 1).float()
            whs_pred = out[1].permute(0, 2, 3, 1).float()
            offsets_pred = out[2].permute(0, 2, 3, 1).float()

            prediction = {
                "hms" : hms_pred,
                "whs" : whs_pred,
                "offsets" : offsets_pred,
                "roi_logits" : out[3].float() if len(out) > 3 else None,
            }

            groundTrue = {
                "hms" : heat_map,
                "whs" : wh,
                "offsets" : offset,
                "masks" : offset_mask,
                "roi_labels" : roi_labels,
            }

            # ponytail: no mIOU here. It is a python-loop decode + NMS per batch and nothing
            # consumed the train-side number. Track it in evaluate() instead.
            loss = criterion(prediction, groundTrue)
            lr = get_lr(optimizer)

            losses.append(loss.item())

            scheduler.step()
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loader.set_postfix(loss=np.mean(losses), lr=lr)
    return float(np.mean(losses))


def evaluate(mode,
             model,
             dataloader,
             criterion,
             DEVICE,
             image_size=(512, 512),
             conf=0.02,
             nms_thres=0.45,
             amp=False,
             num_classes=None,
             roi_topk=3):
    """
    Returns (mean loss, stage-1 mAP@0.5, reranked mAP@0.5, DetectionEval for the rerank).

    conf defaults to 0.02, not the 0.4 used for a demo image. AP integrates the whole
    PR curve, so cutting low-scoring detections simply deletes the high-recall end of
    it and understates the score.
    """
    model.eval()
    net = model.module if hasattr(model, "module") else model
    num_classes = num_classes or net.head.cls_head[-2].out_channels
    losses = []
    ev1 = DetectionEval(num_classes)
    ev2 = DetectionEval(num_classes)

    with torch.no_grad():
        with tqdm(dataloader, ascii=' =', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as loader:
            for image, heat_map, wh, offset, offset_mask, boxes, labels in loader:
                loader.set_description(f"{mode}")

                image = image.to(DEVICE)
                heat_map = heat_map.to(DEVICE)
                wh = wh.to(DEVICE)
                offset = offset.to(DEVICE)
                offset_mask = offset_mask.to(DEVICE)

                gt_rois = [b.to(DEVICE) for b in boxes]
                roi_labels = torch.cat(labels).to(DEVICE) if len(labels) else None

                with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=amp):
                    out = model(image, gt_rois)

                hms_pred = out[0].permute(0, 2, 3, 1).float()
                whs_pred = out[1].permute(0, 2, 3, 1).float()
                offsets_pred = out[2].permute(0, 2, 3, 1).float()

                loss = criterion({"hms": hms_pred, "whs": whs_pred, "offsets": offsets_pred,
                                  "roi_logits": out[3].float() if len(out) > 3 else None},
                                 {"hms": heat_map, "whs": wh, "offsets": offset,
                                  "masks": offset_mask, "roi_labels": roi_labels})
                losses.append(loss.item())

                dets = decode_bbox(
                    postprocess_output(hms_pred, whs_pred, offsets_pred, conf, DEVICE),
                    image_size, DEVICE, image_shape=image_size,
                    remove_pad=True, need_nms=True, nms_thres=nms_thres)

                # ponytail: second forward pass to score the predicted boxes. Eval only,
                # and it keeps the model's forward signature simple. Cache the decoder
                # feature map here if eval time ever matters.
                roi_logits = None
                if net.roi_head is not None:
                    pred_rois = [d[:, :4].float() if len(d) else torch.zeros((0, 4), device=DEVICE)
                                 for d in dets]
                    with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=amp):
                        roi_logits = model(image, pred_rois)[3].float()

                at = 0
                for b in range(len(dets)):
                    gt = torch.cat([boxes[b], labels[b].float().unsqueeze(-1)], -1).cpu().numpy() \
                        if len(labels[b]) else np.zeros((0, 5), np.float32)
                    d = dets[b] if len(dets[b]) else torch.zeros((0, 6), device=DEVICE)
                    ev1.update(d.cpu().numpy(), gt)

                    if roi_logits is not None:
                        n = len(d)
                        ev2.update(roi_rerank(d, roi_logits[at:at + n], roi_topk).cpu().numpy(), gt)
                        at += n

                loader.set_postfix(loss=np.mean(losses), mAP=ev1.mean_ap())

    map1 = ev1.mean_ap()
    map2 = ev2.mean_ap() if net.roi_head is not None else map1
    return float(np.mean(losses)), map1, map2, (ev2 if net.roi_head is not None else ev1)
