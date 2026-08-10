"""Smallest check that the box encode/decode pipeline agrees with itself.

    python test_pipeline.py

If encode_targets, postprocess_output or decode_bbox drift apart (wrong coordinate
format, swapped w/h, lost sub-pixel offset), this fails.
"""
import random

import numpy as np
import torch

from pt_dataset.dataUtils import encode_targets
from utils.detect import postprocess_output, decode_bbox

INPUT_SHAPE = (512, 512)   # (w, h)
STRIDE = 4
OUTPUT_SHAPE = (INPUT_SHAPE[0] // STRIDE, INPUT_SHAPE[1] // STRIDE)


def decode(heat_map, wh, offset, num_classes):
    to_batch = lambda a: torch.from_numpy(a).unsqueeze(0)
    dets = postprocess_output(to_batch(heat_map), to_batch(wh), to_batch(offset), 0.99, "cpu")
    return decode_bbox(dets, INPUT_SHAPE, "cpu", image_shape=INPUT_SHAPE,
                       remove_pad=True, need_nms=True, nms_thres=0.45)[0]


def test_roundtrip():
    # x1, y1, x2, y2 in model input pixels, deliberately odd so sub-pixel centers matter
    boxes = [[100, 120, 181, 201], [300, 60, 421, 141]]
    labels = [0, 0]

    heat_map, wh, offset, mask = encode_targets(boxes, labels, OUTPUT_SHAPE, 1, STRIDE)

    assert mask.sum() == len(boxes), f"expected {len(boxes)} centers, got {mask.sum()}"
    assert heat_map.max() == 1.0, "every box must own one peak of exactly 1.0"

    got = decode(heat_map, wh, offset, 1).numpy()
    assert len(got) == len(boxes), f"decoded {len(got)} boxes, expected {len(boxes)}"

    want = np.array(sorted(boxes, key=lambda b: b[0]))
    got = np.array(sorted(got[:, :4].tolist(), key=lambda b: b[0]))
    err = np.abs(got - want).max()
    assert err <= STRIDE, f"round-trip drift {err:.2f}px exceeds one stride\n{got}\nvs\n{want}"


def test_topk_decode_keeps_both_classes():
    """The whole point of the topk decode: one cell may answer with several classes."""
    boxes = [[100, 100, 180, 180]]
    hm_a, wh, offset, _ = encode_targets(boxes, [0], OUTPUT_SHAPE, 2, STRIDE)
    hm_b, _, _, _ = encode_targets(boxes, [1], OUTPUT_SHAPE, 2, STRIDE)
    heat_map = np.maximum(hm_a, hm_b)

    got = decode(heat_map, wh, offset, 2)
    assert len(got) == 2, f"argmax regression: one cell emitted {len(got)} classes, want 2"
    assert set(got[:, 5].tolist()) == {0.0, 1.0}, f"wrong classes: {got[:, 5]}"


def test_augment_moves_boxes_with_the_image():
    """The augmenter must transform boxes exactly like it transforms pixels."""
    from pt_dataset.dataUtils import Augment

    image = np.zeros((512, 512, 3), np.uint8)
    image[120:200, 100:180] = 255           # one white square, box == the square
    box = [100.0, 120.0, 180.0, 200.0, 0]

    rng_hits = 0
    for seed in range(30):
        random.seed(seed)
        out, boxes = Augment()(image.copy(), [list(box)])
        x1, y1, x2, y2 = (int(round(v)) for v in boxes[0][:4])
        if not (0 <= x1 < x2 <= 512 and 0 <= y1 < y2 <= 512):
            continue                        # affine pushed it off-frame, bbox_check drops it
        rng_hits += 1
        # The white pixels have to be inside the reported box, give or take
        # interpolation and blur at the edges.
        ys, xs = np.nonzero(out[..., 0] > 127)
        assert xs.min() >= x1 - 3 and xs.max() <= x2 + 3, f"seed {seed}: x drift {xs.min()},{xs.max()} vs {x1},{x2}"
        assert ys.min() >= y1 - 3 and ys.max() <= y2 + 3, f"seed {seed}: y drift {ys.min()},{ys.max()} vs {y1},{y2}"
    assert rng_hits > 20, f"only {rng_hits}/30 augmentations kept the box in frame"


def test_rare_classes_stay_in_train():
    from utils.tool import stratified_split

    # class 9 has 3 boxes (below the default threshold), class 0 has 200 (above it).
    # Every rare image also carries a common one, so this also checks that pinning
    # those images to train does not starve the common class's quota.
    ann = [{"path": f"{i}.jpg", "bbox": [[0, 0, 10, 10]], "labels": [0]} for i in range(200)]
    for i in range(3):
        ann[i]["bbox"].append([20, 20, 30, 30])
        ann[i]["labels"].append(9)

    train, valid = stratified_split(ann, valid_ratio=0.2, save=False)
    assert not any(9 in a["labels"] for a in valid), "a rare class leaked into valid"
    assert sum(9 in a["labels"] for a in train) == 3, "rare class lost images"
    assert len(valid) == 40, f"common class quota collapsed: {len(valid)} valid images"


def test_map_is_one_for_perfect_predictions():
    from utils.metrics import DetectionEval

    gt = np.array([[10, 10, 50, 50, 0], [60, 60, 90, 90, 1]], np.float32)
    perfect = np.array([[10, 10, 50, 50, 0.9, 0], [60, 60, 90, 90, 0.8, 1]], np.float32)

    ev = DetectionEval(3)
    ev.update(perfect, gt)
    assert abs(ev.mean_ap() - 1.0) < 1e-6, f"perfect predictions scored {ev.mean_ap()}"
    # class 2 never appears in the ground truth and must not be averaged in as a zero.
    assert np.isnan(ev.per_class_ap()[2]), "absent class was scored instead of skipped"

    # Right box, wrong class: one false positive and one miss, so AP collapses.
    ev = DetectionEval(3)
    ev.update(np.array([[10, 10, 50, 50, 0.9, 1]], np.float32), gt[:1])
    assert ev.mean_ap() == 0.0, f"wrong class still scored {ev.mean_ap()}"

    # Half the ground truth found, at perfect precision -> AP 0.5.
    ev = DetectionEval(1)
    ev.update(np.array([[10, 10, 50, 50, 0.9, 0]], np.float32),
              np.array([[10, 10, 50, 50, 0], [200, 200, 240, 240, 0]], np.float32))
    assert abs(ev.mean_ap() - 0.5) < 1e-6, f"expected 0.5, got {ev.mean_ap()}"


def test_ground_truth_targets_score_perfect_map():
    """Encode boxes, decode them back, score them: the whole eval path must return 1.0.

    This is the check that tells a broken pipeline apart from an undertrained model.
    If a real run reports mAP 0, run this first - it fails only if encode_targets,
    postprocess_output, decode_bbox and DetectionEval have drifted apart.
    """
    from utils.metrics import DetectionEval

    boxes = [[100.0, 120.0, 181.0, 201.0], [300.0, 60.0, 421.0, 141.0], [40.0, 400.0, 96.0, 470.0]]
    labels = [0, 2, 2]
    heat_map, wh, offset, _ = encode_targets(boxes, labels, OUTPUT_SHAPE, 3, STRIDE)

    got = decode(heat_map, wh, offset, 3).numpy()
    gt = np.array([b + [l] for b, l in zip(boxes, labels)], np.float32)

    ev = DetectionEval(3)
    ev.update(got, gt)
    assert abs(ev.mean_ap() - 1.0) < 1e-6, f"eval path scored {ev.mean_ap():.4f} on its own targets"
    assert ev.cm[:3, 3].sum() == 0, "a ground-truth box was reported as missed"


def test_confusion_separates_group_from_background():
    from utils.metrics import DetectionEval

    names = ["plastic_bottle", "glass_bottle", "straw"]      # first two share a group
    gt = np.array([[10, 10, 50, 50, 0]], np.float32)

    ev = DetectionEval(3)
    ev.update(np.array([[10, 10, 50, 50, 0.9, 1]], np.float32), gt)   # called a sibling
    assert ev.cm[0, 1] == 1, "within-group confusion not recorded"
    assert "100% of naming errors stay inside the semantic group" in ev.report(names)

    ev = DetectionEval(3)
    ev.update(np.zeros((0, 6), np.float32), gt)                       # found nothing
    assert ev.cm[0, 3] == 1, "miss not recorded"

    ev = DetectionEval(3)
    ev.update(np.array([[300, 300, 340, 340, 0.9, 2]], np.float32), gt)
    assert ev.cm[3, 2] == 1, "background false positive not recorded"


def test_images_load_in_the_annotation_frame():
    """Decoded image size must equal the size the json claims, for every image.

    360 of 15127 images carry an EXIF rotation. cv2 applies it by default, which
    transposes the image while the boxes stay in the json's frame: 309 of the 1005
    boxes on those images then land outside the picture and the rest sit on the wrong
    pixels. Silent for most of them - it only crashed because aug_retangle indexes a
    corner directly. Skipped when the dataset is not present.
    """
    import json
    import os

    path = "data/train_dataset/train_label.json"
    if not os.path.exists(path):
        print("  (skipped, no dataset)")
        return

    from pt_dataset.dataUtils import read_image_rgb

    with open(path) as f:
        images = json.load(f)["images"]

    root = os.path.join(os.path.dirname(path), "images")
    rng = random.Random(0)
    bad = []
    for im in rng.sample(images, min(150, len(images))):
        p = os.path.join(root, im["filename"])
        if not os.path.exists(p):
            continue
        got = read_image_rgb(p)
        if (got.shape[1], got.shape[0]) != (im["width"], im["height"]):
            bad.append((im["filename"], (im["width"], im["height"]), got.shape[1::-1]))
    assert not bad, f"{len(bad)} images decode to a different size than the json says: {bad[:3]}"


def test_aug_retangle_survives_an_out_of_bounds_box():
    """A box outside the image must not crash the augmenter mid-run."""
    from pt_dataset.dataUtils import aug_retangle

    image = np.zeros((100, 80, 3), np.uint8)
    boxes = [[10.0, 10.0, 30.0, 30.0, 0], [200.0, 5.0, 260.0, 40.0, 1], [-5.0, -5.0, 20.0, 20.0, 2]]
    out, kept = aug_retangle(image, boxes, num_mask=2)
    assert out.shape == image.shape
    assert len(kept) == 1, f"expected 2 masked away, {len(kept)} left"


def test_roi_head_shapes_and_rerank():
    import torch as t
    from model.centerNet import CenterNet
    from utils.detect import roi_rerank

    m = CenterNet(num_classes=5).eval()
    x = t.rand(2, 3, 128, 128)
    rois = [t.tensor([[10., 10., 60., 60.], [20., 20., 80., 80.]]), t.zeros((0, 4))]

    with t.no_grad():
        assert len(m(x)) == 3, "rois=None must keep the 3-output contract for tracing"
        out = m(x, rois)
    assert len(out) == 4 and out[3].shape == (2, 5), f"roi logits {out[3].shape}, want (2, 5)"

    # An image with no boxes must not break the batch.
    with t.no_grad():
        assert m(x, [t.zeros((0, 4)), t.zeros((0, 4))])[3].shape == (0, 5)

    dets = t.tensor([[0., 0., 10., 10., 0.5, 0]])
    logits = t.tensor([[3.0, 2.0, 0.0, 0.0, 0.0]])
    out = roi_rerank(dets, logits, topk=2)
    assert out.shape == (2, 6), f"rerank gave {out.shape}, want one row per hypothesis"
    assert out[0, 5] == 0 and out[1, 5] == 1, "hypotheses not ordered by probability"
    assert out[:, 4].max() <= 0.5, "rerank score exceeded the stage-1 objectness"
    assert (out[0, :4] == dets[0, :4]).all(), "rerank moved the box"


def test_every_backbone_gives_four_levels_at_stride_4_8_16_32():
    """
    A backbone with the wrong taps does not raise, it mis-trains.

    pt_dataset/dataset.py encodes targets at stride 4 and RoIClassifier pools at
    spatial_scale 1/4, both hardcoded. A backbone whose finest level is stride 8 still
    produces a heatmap, just one that disagrees with its targets by a factor of two --
    which is what the old EfficientNet taps {3,5,7,8} did (strides 8/16/32/32). Built
    with weights=None so this test downloads nothing.
    """
    import torch as t
    from model.backnone import BACKBONES, build_backbone

    size = 128
    x = t.rand(1, 3, size, size)
    for name in BACKBONES:
        net = build_backbone(name, weights=None).eval()
        with t.no_grad():
            outs = net(x)
        assert len(outs) == 4, f"{name} returned {len(outs)} levels, want 4"
        strides = [size // o.shape[-1] for o in outs]
        assert strides == [4, 8, 16, 32], f"{name} strides {strides}, want [4, 8, 16, 32]"
        widths = tuple(o.shape[1] for o in outs)
        assert widths == tuple(net.out_channels), \
            f"{name} emits {widths} but declares out_channels={tuple(net.out_channels)}"


def test_autocast_dtype_survives_the_unnormalised_decoder():
    """
    The head BatchNorms are updated in the forward pass, where GradScaler cannot reach
    them, so an overflow in the decoder's unnormalised four-level sum poisons
    running_mean / running_var permanently - and because train() uses batch statistics,
    it stays invisible until eval(). Guard the property that prevents it: the autocast
    dtype must carry an activation past fp16's 65504 ceiling.
    """
    import torch as t
    from utils.pytorchtools import amp_dtype, assert_finite
    from model.centerNet import CenterNet

    assert amp_dtype(t.device("cpu")) is t.float16
    if t.cuda.is_available():
        dev = t.device("cuda:0")
        want = t.bfloat16 if t.cuda.is_bf16_supported() else t.float16
        assert amp_dtype(dev) is want, f"amp_dtype gave {amp_dtype(dev)}, want {want}"
    else:
        return      # the rest needs a GPU to autocast on

    if not t.cuda.is_bf16_supported():
        return      # fp16-only hardware: use_amp is expected to be False there

    m = CenterNet(num_classes=5).to(dev).train()
    # Drive the decoder past fp16's ceiling deliberately. This is the failure the run
    # actually hit, just reached in one step instead of a few hundred.
    with t.no_grad():
        m.decoder.output.weight.mul_(3000.0)

    x = t.rand(2, 3, 128, 128, device=dev)
    with t.autocast(dev.type, dtype=amp_dtype(dev), enabled=True):
        feat = m.decoder(*m.backbone(x))
        peak = feat.abs().max().item()
        assert peak > 65504, f"decoder peaked at {peak:.0f}, below fp16 max - raise the gain"
        m.head(feat)

    assert_finite(m)


if __name__ == "__main__":
    test_roundtrip()
    test_topk_decode_keeps_both_classes()
    test_augment_moves_boxes_with_the_image()
    test_rare_classes_stay_in_train()
    test_map_is_one_for_perfect_predictions()
    test_ground_truth_targets_score_perfect_map()
    test_confusion_separates_group_from_background()
    test_images_load_in_the_annotation_frame()
    test_aug_retangle_survives_an_out_of_bounds_box()
    test_roi_head_shapes_and_rerank()
    test_every_backbone_gives_four_levels_at_stride_4_8_16_32()
    test_autocast_dtype_survives_the_unnormalised_decoder()
    print("ok")
