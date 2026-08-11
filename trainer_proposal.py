"""Train a class-agnostic MobileNetV3 CenterNet proposal detector."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from flow.flow import train
from model.backnone import build_backbone
from model.centerNet import CenterNet
from model.loss import TotalLoss
from pt_dataset.dataloader import dataloader
from utils.detect import decode_bbox, postprocess_output
from utils.metrics import DetectionEval, iou_matrix
from utils.pytorchtools import (
    CosineDecayWarmup,
    EarlyStopping,
    amp_dtype,
    assert_finite,
    get_device,
    load_checkpoint,
    make_scaler,
    save_checkpoint,
)
from utils.tool import collapse_annotations, load_annotation, stratified_split


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", default="data/train_dataset/train_label.json")
    parser.add_argument("--output-dir", default="savemodel_stage2_proposal")
    parser.add_argument("--backbone", default="mobilenet_v3_large")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def evaluate_proposals(model, loader, criterion, device, amp):
    model.eval()
    losses = []
    evaluator = DetectionEval(1)
    hit50 = hit75 = total = 0
    for image, heat_map, wh, offset, offset_mask, boxes, _ in tqdm(
            loader, desc="proposal valid", ascii=" ="):
        image = image.to(device)
        heat_map, wh = heat_map.to(device), wh.to(device)
        offset, offset_mask = offset.to(device), offset_mask.to(device)
        with torch.autocast(device.type, dtype=amp_dtype(device), enabled=amp):
            output = model(image)
        hms, whs, offsets = [value.permute(0, 2, 3, 1).float()
                              for value in output[:3]]
        loss = criterion(
            {"hms": hms, "whs": whs, "offsets": offsets, "roi_logits": None},
            {"hms": heat_map, "whs": wh, "offsets": offset,
             "masks": offset_mask, "roi_labels": None})
        losses.append(loss.item())
        predictions = decode_bbox(
            postprocess_output(hms, whs, offsets, 0.001, device, topk=100),
            (512, 512), device, need_nms=True, nms_thres=0.7)
        for proposals, gt_boxes in zip(predictions, boxes):
            proposals = (proposals.cpu().numpy() if len(proposals)
                         else np.zeros((0, 6), np.float32))
            gt_boxes = gt_boxes.numpy()
            gt = np.concatenate(
                [gt_boxes, np.zeros((len(gt_boxes), 1), np.float32)], axis=1)
            evaluator.update(proposals, gt)
            total += len(gt_boxes)
            if len(gt_boxes) and len(proposals):
                best_iou = iou_matrix(gt_boxes, proposals[:50, :4]).max(1)
                hit50 += int((best_iou >= 0.5).sum())
                hit75 += int((best_iou >= 0.75).sum())
    return {
        "loss": float(np.mean(losses)),
        "map50": evaluator.mean_ap(),
        "recall50": hit50 / max(1, total),
        "recall75": hit75 / max(1, total),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations, _ = load_annotation(args.annotation)
    train_annotations, valid_annotations = stratified_split(
        annotations, valid_ratio=0.2, seed=42, save=False)
    train_annotations = collapse_annotations(train_annotations)
    valid_annotations = collapse_annotations(valid_annotations)
    train_loader, valid_loader = dataloader(
        train_annotations, valid_annotations, 1, args.batch_size, (512, 512),
        num_workers=args.workers, cache=args.cache_images, rfs_thresh=0.0,
        mosaic_p=0.2, cutout_p=0.0, gauss_min_overlap=0.7)

    model = CenterNet(
        num_classes=1, backbone=build_backbone(args.backbone), roi_head=False).to(device)
    criterion = TotalLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scaler = make_scaler(device, enabled=not args.no_amp)
    scheduler = CosineDecayWarmup(
        optimizer, args.lr, warmup_len=3 * len(train_loader),
        total_iters=args.epochs * len(train_loader), min_lr=1e-6)
    early_stopping = EarlyStopping(patience=args.epochs + 1)
    start_epoch = 0
    best_score = 0.0
    train_losses, valid_losses, map_history = [], [], []
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_score, train_losses, valid_losses, map_history = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, early_stopping,
            map_location=device)

    for epoch in range(start_epoch, args.epochs):
        if epoch >= int(args.epochs * 0.75) and train_loader.dataset.mosaic_p:
            train_loader.dataset.close_augment()
        train_loss = train(
            epoch, model, optimizer, scheduler, train_loader, criterion, device, scaler)
        assert_finite(model)
        metrics = evaluate_proposals(
            model, valid_loader, criterion, device, scaler.is_enabled())
        print(f"epoch {epoch}: loss {metrics['loss']:.4f}  AP {metrics['map50']:.4f}  "
              f"R@50 IoU.5 {metrics['recall50']:.4f}  IoU.75 {metrics['recall75']:.4f}")
        train_losses.append(train_loss)
        valid_losses.append(metrics["loss"])
        map_history.append(metrics["map50"])
        score = metrics["recall50"] + 1e-3 * metrics["recall75"]
        if score >= best_score:
            best_score = score
            torch.save(model.state_dict(), output_dir / "model.pth")
        save_checkpoint(
            str(output_dir / "last.pth"), epoch, model, optimizer, scheduler, scaler,
            early_stopping, best_score, train_losses, valid_losses, map_history)


if __name__ == "__main__":
    main()
