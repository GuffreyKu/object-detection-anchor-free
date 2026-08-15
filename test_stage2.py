"""Small offline checks for the added two-stage path."""

import numpy as np
import torch

from pt_dataset.dataUtils import crop_and_letterbox, crop_centered_or_letterbox
from model.loss import HardNegativeCrossEntropy
from utils.detect import class_agnostic_nms, classifier_detections
from utils.metrics import proposal_recall
from utils.tool import (
    collapse_annotations,
    crop_sample_weights,
    crop_samples_from_records,
    filter_invalid_proposals,
    select_proposal_topk,
)


def test_crop_and_letterbox_preserves_shape():
    image = np.zeros((40, 80, 3), np.uint8)
    image[10:30, 20:60] = (10, 20, 30)
    crop = crop_and_letterbox(image, [20, 10, 60, 30], size=64, expand=0)
    assert crop.shape == (64, 64, 3)
    assert (crop[0] == 114).all() and (crop[-1] == 114).all()
    assert tuple(crop[32, 32]) == (10, 20, 30)


def test_small_box_uses_native_scale_center_crop():
    image = np.zeros((300, 300, 3), np.uint8)
    image[140:160, 140:160] = 255
    crop = crop_centered_or_letterbox(image, [140, 140, 160, 160])
    assert crop.shape == (224, 224, 3)
    assert (crop[102:122, 102:122] == 255).all()
    assert (crop[..., 0] == 255).sum() == 20 * 20

    edge = crop_centered_or_letterbox(image, [0, 0, 20, 20])
    assert (edge[0, 0] == 114).all()


def test_class_agnostic_nms_merges_classes():
    detections = torch.tensor([
        [10., 0., 0., 10., .95, 1.],
        [0., 0., 10., 10., .9, 0.],
        [0., 0., 10., 10., .8, 7.],
        [20., 20., 30., 30., .7, 2.],
    ])
    proposals = class_agnostic_nms(detections, 0.7, 100)
    assert proposals.shape == (2, 5)
    assert torch.allclose(proposals[:, 4], torch.tensor([0.9, 0.7]))


def test_invalid_cached_proposals_are_removed():
    records = [{"proposals": [
        [569.52, 452.05, 569.43, 571.88, .9],
        [0, 0, 10, 10, .8],
    ]}]
    assert filter_invalid_proposals(records)[0]["proposals"] == [[0, 0, 10, 10, .8]]


def test_classifier_score_does_not_multiply_objectness():
    proposals = [[0, 0, 10, 10, 0.01]]
    probabilities = np.zeros((1, 35), np.float32)
    probabilities[0, 4] = 0.8
    probabilities[0, 34] = 0.2
    detections = classifier_detections(proposals, probabilities)
    assert detections.shape == (1, 6)
    assert np.isclose(detections[0, 4].item(), probabilities[0, 4])
    assert detections[0, 5].item() == 4


def test_crop_sample_iou_bands_and_background_mass():
    records = [{
        "path": "unused.jpg",
        "gt_boxes": [[0, 0, 10, 10]],
        "gt_labels": [3],
        "proposals": [
            [0, 0, 10, 10, .9],       # positive
            [4, 0, 14, 10, .8],       # IoU .43: ignored
            [20, 20, 30, 30, .7],     # background
        ],
    }]
    samples = crop_samples_from_records(records)
    labels = [sample["label"] for sample in samples]
    assert labels.count(3) == 2       # GT + matched detector crop
    assert labels.count(34) == 1
    weights = crop_sample_weights(samples)
    background_mass = weights[np.array(labels) == 34].sum() / weights.sum()
    assert abs(background_mass.item() - 0.25) < 1e-9


def test_proposal_recall_and_safe_topk():
    records = [{
        "gt_boxes": [[0, 0, 10, 10]],
        "proposals": [[20, 20, 30, 30, .9], [0, 0, 10, 10, .8]],
    }]
    assert proposal_recall(records, 0.5, 1) == 0
    assert proposal_recall(records, 0.5, 2) == 1
    assert select_proposal_topk(records, candidates=(1, 2)) == 2
    empty = [{"gt_boxes": [[0, 0, 10, 10]], "proposals": []}]
    assert select_proposal_topk(empty, candidates=(10, 100)) == 100


def test_collapse_annotations_does_not_mutate_input():
    source = [{"path": "x", "bbox": [[0, 0, 1, 1]], "labels": [7]}]
    collapsed = collapse_annotations(source)
    assert source[0]["labels"] == [7]
    assert collapsed[0]["labels"] == [0]


def test_hard_negative_loss_uses_same_group_hardest_logit():
    logits = torch.tensor([[1.0, 2.0, -2.0, 3.0]], requires_grad=True)
    target = torch.tensor([0])
    criterion = HardNegativeCrossEntropy(
        4, hard_negative_groups=[[0, 1, 2]], hard_weight=0.5,
        margin=0.2, label_smoothing=0)
    base = torch.nn.functional.cross_entropy(logits, target)
    loss = criterion(logits, target)
    expected_margin = torch.relu(logits[0, 1] + 0.2 - logits[0, 0])
    assert torch.allclose(loss, base + 0.5 * expected_margin)
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0


if __name__ == "__main__":
    test_crop_and_letterbox_preserves_shape()
    test_small_box_uses_native_scale_center_crop()
    test_class_agnostic_nms_merges_classes()
    test_invalid_cached_proposals_are_removed()
    test_classifier_score_does_not_multiply_objectness()
    test_crop_sample_iou_bands_and_background_mass()
    test_proposal_recall_and_safe_topk()
    test_collapse_annotations_does_not_mutate_input()
    test_hard_negative_loss_uses_same_group_hardest_logit()
    print("ok")
