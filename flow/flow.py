import torch
import numpy as np
from tqdm import tqdm
from utils.detect import postprocess_output, decode_bbox, roi_rerank
from utils.metrics import DetectionEval
from utils.pytorchtools import amp_dtype

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
    dtype = amp_dtype(DEVICE)
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

            with torch.autocast(DEVICE.type, dtype=dtype, enabled=amp):
                out = model(image, rois)

            # Back to fp32 for the loss: focal loss takes log() of a sigmoid and
            # neither reduced dtype has the mantissa for it - fp16 underflows, and
            # bf16's 8 bits are coarser still.
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
             conf=0.001,
             nms_thres=0.45,
             amp=False,
             num_classes=None,
             roi_topk=3,
             topk=1000,
             stage2=False):
    """
    Returns (mean loss, stage-1 mAP@0.5, reranked mAP@0.5 or None, DetectionEval).

    conf is 0.001 and topk 1000, not the 0.1/100 used for a demo image. AP here is
    all-point interpolated, so a detection appended below every existing score either
    is a false positive - which leaves recall unchanged, gets dropped by the
    `mrec[1:] != mrec[:-1]` step filter and cannot touch the backward precision
    envelope - or is a true positive, which raises recall. Lowering conf and raising
    topk therefore cannot decrease AP, only uncover the high-recall tail of the curve.

    stage2 is off. Measured on the run-1 weights over the full validation set, the RoI
    rerank is a regression at every setting: stage-1 alone 0.4474, rerank k=3 0.4023,
    k=1 0.3854, and 0.4331 for rescoring without relabelling. roi_rerank multiplies
    objectness by a softmax over 34 classes, which divides scores by 3-30x and destroys
    the cross-image ordering AP depends on, and it expands every box into `roi_topk`
    rows, so each background box becomes roi_topk background false positives.
    The head is still TRAINED (its cross-entropy shapes the shared decoder features);
    only its use at eval time is off. Turning this back on costs a second full backbone
    forward per batch, and every fusion rule can be tried offline on the saved weights.
    """
    model.eval()
    dtype = amp_dtype(DEVICE)
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

                with torch.autocast(DEVICE.type, dtype=dtype, enabled=amp):
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
                    postprocess_output(hms_pred, whs_pred, offsets_pred, conf, DEVICE,
                                       topk=topk),
                    image_size, DEVICE, image_shape=image_size,
                    remove_pad=True, need_nms=True, nms_thres=nms_thres)

                # ponytail: second forward pass to score the predicted boxes. Eval only,
                # and it keeps the model's forward signature simple. Cache the decoder
                # feature map here if eval time ever matters.
                roi_logits = None
                if stage2 and net.roi_head is not None:
                    pred_rois = [d[:, :4].float() if len(d) else torch.zeros((0, 4), device=DEVICE)
                                 for d in dets]
                    with torch.autocast(DEVICE.type, dtype=dtype, enabled=amp):
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

    # map2 is None when stage 2 did not run, rather than a copy of map1: a caller that
    # prints it as "reranked" should say "off", not repeat the stage-1 number.
    used_stage2 = stage2 and net.roi_head is not None
    map1 = ev1.mean_ap()
    map2 = ev2.mean_ap() if used_stage2 else None
    return float(np.mean(losses)), map1, map2, (ev2 if used_stage2 else ev1)
