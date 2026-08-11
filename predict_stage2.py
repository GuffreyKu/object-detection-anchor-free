"""Run detector -> crop classifier inference over a folder."""

import argparse
import json
import os
from pathlib import Path

import torch

from pt_dataset.dataUtils import read_image_rgb
from utils.detect import classifier_detections
from utils.pytorchtools import get_device, load_crop_classifier_checkpoint
from utils.tool import (
    classify_proposal_records,
    infer_stage2_proposals,
    load_annotation,
    load_stage2_detector,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", default="submission_stage2.json")
    parser.add_argument("--detector-weights", required=True)
    parser.add_argument("--detector-backbone", default="swin_t")
    parser.add_argument("--detector-class-agnostic", action="store_true")
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--annotation", default="data/train_dataset/train_label.json")
    parser.add_argument("--proposal-topk", type=int)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--max-detections", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    _, class_names = load_annotation(args.annotation)
    detector = load_stage2_detector(
        args.detector_weights, args.detector_backbone, len(class_names), device,
        class_agnostic=args.detector_class_agnostic)
    classifier, checkpoint = load_crop_classifier_checkpoint(
        args.classifier, device, expected_class_names=class_names)
    proposal_topk = args.proposal_topk or checkpoint.get("proposal_topk", 100)
    files = sorted(filename for filename in os.listdir(args.images)
                   if filename.lower().endswith(IMAGE_EXTENSIONS))
    if not files:
        raise SystemExit(f"no images under {args.images}")

    results = []
    with torch.no_grad():
        for index, filename in enumerate(files):
            path = os.path.join(args.images, filename)
            proposals, _ = infer_stage2_proposals(
                detector, read_image_rgb(path), device, max_proposals=proposal_topk)
            record = {"path": path, "proposals": proposals.tolist()}
            probabilities = classify_proposal_records(
                classifier, [record], device, batch_size=128, workers=0,
                size=checkpoint["input_size"], expand=checkpoint["crop_expand"])[0]
            detections = classifier_detections(
                proposals, probabilities, len(class_names), args.score_threshold,
                0.45, args.max_detections).numpy()
            for x1, y1, x2, y2, score, label in detections:
                label = int(label)
                results.append({
                    "image_id": os.path.splitext(filename)[0],
                    "category_id": label,
                    "category_name": class_names[label],
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                    "score": round(float(score), 5),
                })
            if index % 200 == 0:
                print(f"{index}/{len(files)}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results))
    print(f"wrote {output}: {len(results)} detections over {len(files)} images")


if __name__ == "__main__":
    main()
