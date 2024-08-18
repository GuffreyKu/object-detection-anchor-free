import torch
import numpy as np
from tqdm import tqdm
from utils.detect import postprocess_output, decode_bbox

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def compute_iou(box1, box2):
    # box1: shape (num_gt_boxes, 1, 4)
    # box2: shape (1, num_pd_boxes, 4)
    
    # Compute the coordinates of the intersection rectangle
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    # Compute the width and height of the intersection rectangle
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)

    # Compute the area of the intersection rectangle
    inter_area = inter_w * inter_h

    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    # Compute the area of the union
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / torch.clamp(union_area, min=1e-6)  # Avoid division by zero

    return iou

def mIOU(prediction, groundTrue, image_size, conf, nms_thres, dev):
    pred_decode = postprocess_output(prediction["hms"], prediction["whs"], prediction["offsets"], conf, dev)
    pred_decode = decode_bbox(pred_decode, image_size, dev, image_shape=image_size, remove_pad=True, need_nms=True, nms_thres=nms_thres)
    
    gt_decode = postprocess_output(groundTrue["hms"], groundTrue["whs"], groundTrue["offsets"], 0.99, dev)
    gt_decode = decode_bbox(gt_decode, image_size, dev, image_shape=image_size, remove_pad=True, need_nms=True, nms_thres=nms_thres)

    ious = []
    # Iterate over each ground truth box
    for i in range(len(gt_decode)):
        if (len(pred_decode[i]) > 0) and (len(gt_decode[i]) > 0):
            gt_boxes = gt_decode[i][:, :4].unsqueeze(1) 
            pd_boxes = pred_decode[i][:, :4].unsqueeze(0)

            ious_matrix = compute_iou(gt_boxes, pd_boxes)  # shape (num_gt_boxes, num_pd_boxes)

            # Filter IoUs greater than 0 and append to the ious list
            valid_ious = ious_matrix[ious_matrix >= 0].cpu().detach().numpy().tolist()
            ious.extend(valid_ious)
    # Calculate mean IOU
    if len(ious) > 0:
        mean_iou = np.mean(ious)
    else:
        mean_iou = 0

    return mean_iou

def train(now_ep,
          model,
          optimizer,
          scheduler,
          dataloader,
          criterion,
          DEVICE):
    
    losses = []
    mious = []
    model.train()
    with tqdm(dataloader, ascii=' =', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as loader:
        for image, heat_map, wh, offset, offset_mask in loader:
            loader.set_description(f"train {now_ep}")
            image = image.to(DEVICE)

            heat_map = heat_map.to(DEVICE)
            wh = wh.to(DEVICE)
            offset = offset.to(DEVICE)
            offset_mask = offset_mask.to(DEVICE)

            optimizer.zero_grad()

            hms_pred, whs_pred, offsets_pred = model(image)
            hms_pred = hms_pred.permute(0, 2, 3, 1)
            whs_pred = whs_pred.permute(0, 2, 3, 1)
            offsets_pred = offsets_pred.permute(0, 2, 3, 1)
 
            prediction = {
                "hms" : hms_pred,
                "whs" : whs_pred,
                "offsets" : offsets_pred
            }

            groundTrue = {
                "hms" : heat_map,
                "whs" : wh,
                "offsets" : offset,
                "masks" : offset_mask
            }

            miou = mIOU(prediction, groundTrue, (512, 512), 0.4, 0.45, DEVICE)
            
            loss = criterion(prediction, groundTrue)
            lr = get_lr(optimizer)

            mious.append(miou)
            losses.append(loss.item())

            loss.backward()
            scheduler.step()
            optimizer.step()

            loader.set_postfix(loss=np.mean(losses), miou=np.mean(mious), lr=lr)
    return np.mean(losses), miou

def evaluate(mode,
             model,
             dataloader,
             criterion,
             DEVICE):
    
    model.eval()
    losses = []
    mious = []
    with torch.no_grad():
        with tqdm(dataloader, ascii=' =', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as loader:
            for image, heat_map, wh, offset, offset_mask in loader:
                loader.set_description(f"{mode}")

                image = image.to(DEVICE)

                heat_map = heat_map.to(DEVICE)
                wh = wh.to(DEVICE)
                offset = offset.to(DEVICE)
                offset_mask = offset_mask.to(DEVICE)
                

                hms_pred, whs_pred, offsets_pred = model(image)

                hms_pred = hms_pred.permute(0, 2, 3, 1)
                whs_pred = whs_pred.permute(0, 2, 3, 1)
                offsets_pred = offsets_pred.permute(0, 2, 3, 1)
                
                prediction = {
                    "hms" : hms_pred,
                    "whs" : whs_pred,
                    "offsets" : offsets_pred
                }

                groundTrue = {
                    "hms" : heat_map,
                    "whs" : wh,
                    "offsets" : offset,
                    "masks" : offset_mask
                }
                miou = mIOU(prediction, groundTrue, (512, 512), 0.4, 0.45, DEVICE)
                loss = criterion(prediction, groundTrue)
                losses.append(loss.item())
                mious.append(miou)
                
                loader.set_postfix(loss=np.mean(losses), miou=np.mean(mious))
    return np.mean(losses), miou
