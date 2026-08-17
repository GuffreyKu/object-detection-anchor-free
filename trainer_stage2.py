"""Train a separate 34-class + background crop classifier from detector proposals.

Example:
    python trainer_stage2.py --detector-weights /path/to/model.pth --detector-backbone swin_t
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from model.centerNet import CropClassifier
from model.loss import HardNegativeCrossEntropy, inverse_sqrt_class_weights
from pt_dataset.dataset import CropDataset
from utils.metrics import GROUPS, proposal_recall
from utils.pytorchtools import (
    EarlyStopping,
    amp_dtype,
    get_device,
    load_crop_classifier_checkpoint,
    make_scaler,
    save_crop_classifier_checkpoint,
)
from utils.tool import (
    crop_sample_weights,
    crop_samples_from_records,
    evaluate_crop_classifier,
    evaluate_stage1_records,
    evaluate_stage2_records,
    gt_crop_samples,
    limit_record_proposals,
    load_annotation,
    load_or_generate_proposal_records,
    load_stage2_detector,
    proposal_cache_metadata,
    select_proposal_topk,
    stratified_split,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-weights", required=True)
    parser.add_argument("--detector-backbone", default="swin_t")
    parser.add_argument("--detector-class-agnostic", action="store_true")
    parser.add_argument("--annotation", default="data/train_dataset/train_label.json")
    parser.add_argument("--output-dir", default="savemodel_stage2_classifier")
    parser.add_argument("--manifest-dir", default="data/stage2_cache")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--hard-negative-weight", type=float, default=0.5)
    parser.add_argument("--hard-negative-margin", type=float, default=0.2)
    parser.add_argument("--rare-loss-cap", type=float, default=4.0)
    parser.add_argument("--patience", type=int, default=8,
                        help="stop after this many epochs with no two-stage mAP "
                             "improvement (docs/STAGE2_EVAL.md: a prior run's mAP "
                             "peaked at epoch 3 of 30 and never improved again)")
    parser.add_argument("--arc-margin", type=float, default=0.0,
                        help="ArcFace additive angular margin in radians on the crop "
                             "classifier head; 0 (default) keeps the plain Linear head "
                             "and prior behaviour. Experimental - see model/loss.py: "
                             "HardNegativeCrossEntropy and docs/STAGE2_EVAL.md")
    parser.add_argument("--arc-scale", type=float, default=30.0,
                        help="ArcFace logit scale, only used when --arc-margin > 0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def cosine_with_warmup(optimizer, total_steps, warmup_steps):
    def scale(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    annotations, class_names = load_annotation(args.annotation)
    train_annotations, valid_annotations = stratified_split(
        annotations, valid_ratio=0.2, seed=42, save=False)
    detector = load_stage2_detector(
        args.detector_weights, args.detector_backbone, len(class_names), device,
        class_agnostic=args.detector_class_agnostic)

    cache_dir = Path(args.manifest_dir)
    common = dict(input_shape=(512, 512), confidence=0.001, topk=1000,
                  proposal_nms=0.7, max_proposals=100)
    train_meta = proposal_cache_metadata(
        args.detector_weights, args.detector_backbone,
        args.detector_class_agnostic, keep_stage1=False, **common)
    valid_meta = proposal_cache_metadata(
        args.detector_weights, args.detector_backbone,
        args.detector_class_agnostic,
        keep_stage1=not args.detector_class_agnostic, **common)
    train_records = load_or_generate_proposal_records(
        cache_dir / "train_records.json", train_annotations, detector, device,
        train_meta, rebuild=args.rebuild_manifest)
    valid_records = load_or_generate_proposal_records(
        cache_dir / "valid_records.json", valid_annotations, detector, device,
        valid_meta, rebuild=args.rebuild_manifest)
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    proposal_topk = select_proposal_topk(valid_records)
    valid_eval_records = limit_record_proposals(valid_records, proposal_topk)
    print(f"proposal topK={proposal_topk}; R@.5="
          f"{proposal_recall(valid_records, 0.5, proposal_topk):.4f}, R@.75="
          f"{proposal_recall(valid_records, 0.75, proposal_topk):.4f}")
    if not args.detector_class_agnostic:
        stage1_map, _ = evaluate_stage1_records(valid_records, len(class_names))
        print(f"cached stage-1 mAP@0.5 {stage1_map:.4f}")

    train_samples = crop_samples_from_records(train_records, background=len(class_names))
    valid_samples = gt_crop_samples(valid_records)
    train_loader = DataLoader(
        CropDataset(train_samples, train=True), batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            crop_sample_weights(train_samples, background=len(class_names)),
            num_samples=len(train_samples), replacement=True),
        num_workers=args.workers, persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda", drop_last=True)
    print(f"crop samples: {len(train_samples)} train / {len(valid_samples)} valid GT")

    checkpoint = None
    if args.resume:
        model, checkpoint = load_crop_classifier_checkpoint(
            args.resume, device, expected_class_names=class_names)
    else:
        model = CropClassifier(
            len(class_names), pretrained=True, arc_margin=args.arc_margin > 0).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = cosine_with_warmup(
        optimizer, args.epochs * len(train_loader), len(train_loader))
    early_stopping = EarlyStopping(patience=args.patience, verbose=False)
    start_epoch, best_map = 0, 0.0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_map = checkpoint.get("best_map", 0.0)
        if checkpoint.get("early_stopping"):
            early_stopping.load_state_dict(checkpoint["early_stopping"])
    counts = np.bincount(
        [sample["label"] for sample in train_samples], minlength=len(class_names) + 1)
    class_weights = inverse_sqrt_class_weights(
        counts, clip=(0.5, args.rare_loss_cap))
    class_weights[len(class_names)] = 1.0
    name_to_index = {name: index for index, name in enumerate(class_names)}
    hard_negative_groups = [
        [name_to_index[name] for name in members if name in name_to_index]
        for members in GROUPS.values()
    ]
    criterion = HardNegativeCrossEntropy(
        len(class_names) + 1, class_weights, hard_negative_groups,
        args.hard_negative_weight, args.hard_negative_margin,
        arc_margin=args.arc_margin, arc_scale=args.arc_scale).to(device)
    rarest = int(counts[:len(class_names)].argmin())
    print(f"hard negatives: weight={args.hard_negative_weight}, "
          f"margin={args.hard_negative_margin}; rarest {class_names[rarest]} "
          f"n={counts[rarest]}, loss weight={class_weights[rarest]:.2f}")
    if args.arc_margin > 0:
        print(f"arc margin ON: margin={args.arc_margin} rad, scale={args.arc_scale} "
              f"- experimental, see docs/STAGE2_EVAL.md")
    scaler = make_scaler(device, enabled=not args.no_amp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        loader = tqdm(train_loader, desc=f"stage2 {epoch}", ascii=" =")
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(
                    device.type, dtype=amp_dtype(device), enabled=scaler.is_enabled()):
                loss = criterion(model(images), labels)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            loader.set_postfix(loss=np.mean(losses), lr=optimizer.param_groups[0]["lr"])

        oracle = evaluate_crop_classifier(
            model, valid_samples, device, args.batch_size, args.workers)
        map50, evaluator, _ = evaluate_stage2_records(
            model, valid_eval_records, device, len(class_names),
            args.batch_size, args.workers)
        print(f"epoch {epoch}: train {np.mean(losses):.4f}  "
              f"oracle acc {oracle['accuracy']:.4f} macro-R {oracle['macro_recall']:.4f}  "
              f"two-stage mAP@0.5 {map50:.4f}")
        improved = map50 >= best_map
        if improved:
            best_map = map50
        save_crop_classifier_checkpoint(
            output_dir / "last.pth", model, optimizer, scheduler, epoch, best_map,
            class_names, proposal_topk=proposal_topk, arc_margin=args.arc_margin > 0,
            early_stopping=early_stopping)
        if improved:
            save_crop_classifier_checkpoint(
                output_dir / "best.pth", model, optimizer, scheduler, epoch, best_map,
                class_names, proposal_topk=proposal_topk, arc_margin=args.arc_margin > 0,
                early_stopping=early_stopping)
            print(evaluator.report(class_names))

        early_stopping(-map50)
        if early_stopping.early_stop:
            print(f"epoch {epoch}: no two-stage mAP improvement in {args.patience} "
                  f"epochs, stopping (best {best_map:.4f})")
            break


if __name__ == "__main__":
    main()
