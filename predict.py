"""Run the trained detector over a folder of images and write a COCO-style result file.

    uv run python predict.py --images data/test_dataset/images --out submission.json

Stage 1 only. Measured on the run-1 weights over the full validation set, the RoI
rerank was a 4.5-point regression at every setting tried, so predict_full()/roi_rerank
are deliberately not used here - see flow.evaluate's docstring.

THE COORDINATE TRANSFORM. Training and inference both resize with
`cv2.resize(image, (512, 512))`, a plain stretch that does not preserve aspect ratio and
adds no padding. `decode_bbox(..., remove_pad=True, image_shape=(W, H))` implements the
LETTERBOX inverse - `scale = min(iw/w, ih/h)` and then subtracting dw/dh - which is the
wrong inverse for this pipeline. Every in-repo caller happens to pass
image_shape == input_shape, so scale is 1 and dw = dh = 0 and the mismatch is invisible.
Feed it a real image size and every box moves. The correct inverse for a stretch is the
per-axis ratio applied below, which is also exactly what ImgDataset.resize_image does to
the boxes in the forward direction.
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from model.centerNet import CenterNet
from pt_dataset.dataUtils import read_image_rgb, to_chw_tensor
from utils.detect import postprocess_output, decode_bbox
from utils.pytorchtools import get_device, load_weights
from utils.tool import load_annotation

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def predict_one(model, path, device, input_shape, conf, nms_thres, topk, tta):
    """Boxes in ORIGINAL image pixels: (N, 6) x1 y1 x2 y2 score label."""
    image = read_image_rgb(path)                     # same EXIF handling as training
    raw_h, raw_w = image.shape[:2]
    resized = cv2.resize(image, input_shape)
    x = torch.from_numpy(to_chw_tensor(resized)).unsqueeze(0).to(device)

    hms, whs, offsets = model(x)
    if tta:
        # Horizontal flip. The heatmap and the size map flip back exactly because the
        # width is a whole number of stride-4 cells; the offset map does not - its value
        # is the sub-cell remainder of the centre, which mirrors to (1 - v) in x and
        # would need its own correction - so the un-flipped offsets are kept as-is.
        f_hms, f_whs, _ = model(torch.flip(x, dims=[3]))
        hms = 0.5 * (hms + torch.flip(f_hms, dims=[3]))
        whs = 0.5 * (whs + torch.flip(f_whs, dims=[3]))

    hms, whs, offsets = [t.permute(0, 2, 3, 1).float() for t in (hms, whs, offsets)]
    dets = decode_bbox(
        postprocess_output(hms, whs, offsets, conf, device, topk=topk),
        input_shape, device, need_nms=True, nms_thres=nms_thres)[0]
    if len(dets) == 0:
        return np.zeros((0, 6), np.float32)

    dets = dets.cpu().numpy()
    # Stretch inverse, per axis. NOT decode_bbox(remove_pad=True) - see the module note.
    dets[:, [0, 2]] *= raw_w / input_shape[0]
    dets[:, [1, 3]] *= raw_h / input_shape[1]
    np.clip(dets[:, [0, 2]], 0, raw_w, out=dets[:, [0, 2]])
    np.clip(dets[:, [1, 3]], 0, raw_h, out=dets[:, [1, 3]])
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of images to predict")
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--weights", default="savemodel/model.pth")
    ap.add_argument("--backbone", default="swin_t")
    ap.add_argument("--labels", default="data/train_dataset/train_label.json",
                    help="only read for the class-name/id mapping")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="low on purpose: AP integrates the whole PR curve")
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--max-per-image", type=int, default=100,
                    help="keep the top-N by score per image. At conf=0.001 the raw "
                         "output is ~750 boxes/image, which is a large file for no "
                         "benefit. Measured on the run-1 weights over the full "
                         "validation set: none 0.4505, 300 0.4501, 100 0.4478, "
                         "50 0.4440. 100 is the COCO maxDets convention and costs "
                         "0.003; use 300 if the platform accepts the bigger file.")
    ap.add_argument("--no-tta", action="store_true", help="disable horizontal-flip TTA")
    args = ap.parse_args()

    device = get_device()
    _, class_names = load_annotation(args.labels)

    model = CenterNet(num_classes=len(class_names), backbone=args.backbone).to(device)
    load_weights(model, torch.load(args.weights, map_location=device), args.weights)
    model.eval()

    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(IMG_EXT))
    if not files:
        raise SystemExit(f"no images under {args.images}")
    print(f"{len(files)} images, {len(class_names)} classes, tta={not args.no_tta}")

    results, n_boxes = [], 0
    with torch.no_grad():
        for i, name in enumerate(files):
            dets = predict_one(model, os.path.join(args.images, name), device,
                               (512, 512), args.conf, args.nms, args.topk,
                               tta=not args.no_tta)
            if args.max_per_image and len(dets) > args.max_per_image:
                dets = dets[np.argsort(-dets[:, 4])[:args.max_per_image]]
            for x1, y1, x2, y2, score, label in dets:
                results.append({
                    "image_id": os.path.splitext(name)[0],
                    "category_id": int(label),
                    "category_name": class_names[int(label)],
                    # COCO wants xywh, not xyxy.
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                    "score": round(float(score), 5),
                })
            n_boxes += len(dets)
            if i % 200 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    with open(args.out, "w") as fh:
        json.dump(results, fh)
    print(f"wrote {args.out}: {n_boxes} boxes over {len(files)} images "
          f"({n_boxes / len(files):.1f}/image)")


if __name__ == "__main__":
    main()
