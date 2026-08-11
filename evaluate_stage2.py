"""Evaluate stage-1 proposals and the separate crop classifier."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from utils.metrics import proposal_recall
from utils.pytorchtools import get_device, load_crop_classifier_checkpoint
from utils.tool import (
    evaluate_crop_classifier,
    evaluate_stage1_records,
    evaluate_stage2_records,
    gt_crop_samples,
    limit_record_proposals,
    load_annotation,
    load_or_generate_proposal_records,
    load_stage2_detector,
    proposal_cache_metadata,
    stratified_split,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-weights", required=True)
    parser.add_argument("--detector-backbone", default="swin_t")
    parser.add_argument("--detector-class-agnostic", action="store_true")
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--annotation", default="data/train_dataset/train_label.json")
    parser.add_argument("--manifest", default="data/stage2_cache/valid_records.json")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    annotations, class_names = load_annotation(args.annotation)
    _, valid_annotations = stratified_split(
        annotations, valid_ratio=0.2, seed=42, save=False)
    detector = load_stage2_detector(
        args.detector_weights, args.detector_backbone, len(class_names), device,
        class_agnostic=args.detector_class_agnostic)
    metadata = proposal_cache_metadata(
        args.detector_weights, args.detector_backbone, args.detector_class_agnostic,
        keep_stage1=not args.detector_class_agnostic)
    records = load_or_generate_proposal_records(
        args.manifest, valid_annotations, detector, device, metadata,
        rebuild=args.rebuild_manifest)
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    classifier, checkpoint = load_crop_classifier_checkpoint(
        args.classifier, device, expected_class_names=class_names)
    proposal_topk = checkpoint.get("proposal_topk", 100)
    oracle = evaluate_crop_classifier(
        classifier, gt_crop_samples(records), device, args.batch_size, args.workers,
        checkpoint["input_size"], checkpoint["crop_expand"])
    map50, evaluator, _ = evaluate_stage2_records(
        classifier, limit_record_proposals(records, proposal_topk), device,
        len(class_names), args.batch_size, args.workers,
        checkpoint["input_size"], checkpoint["crop_expand"])
    result = {
        "proposal_topk": proposal_topk,
        "proposal_recall": {
            f"iou_{iou}_k_{topk}": proposal_recall(records, iou, topk)
            for iou in (0.5, 0.75) for topk in (10, 20, 50, 100)
        },
        "oracle_classifier": oracle,
        "two_stage_map50": map50,
        "per_class_ap": {
            name: None if np.isnan(ap) else float(ap)
            for name, ap in zip(class_names, evaluator.per_class_ap())
        },
    }
    if not args.detector_class_agnostic:
        result["stage1_map50"] = evaluate_stage1_records(records, len(class_names))[0]
    print(json.dumps(result, indent=2))
    print(evaluator.report(class_names))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
