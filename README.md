# object-detection-anchor-free

CenterNet-style anchor-free detector for marine litter, 34 classes.
ResNet50 backbone, FPN-ish decoder, heatmap + wh + offset heads at stride 4, RGB in.

See [docs/DATASET.md](docs/DATASET.md) for the distribution analysis and what it means
for modelling — class imbalance, stride choice, augmentation limits, and the known
blockers on mAP@0.5.

## Dataset
* COCO-style json, images on local disk.
* `data/train_dataset/train_label.json` plus `data/train_dataset/images/`.
* `stratified_split()` holds out ~20% per class, except classes under 150 boxes,
  which stay entirely in train (see DATASET.md §8).

## Install
```
pip install -r requirements.txt
```
`tensorrt` is not in requirements, install the build matching your CUDA if you need `onnx2trt.py`.

## Hardware
`get_device()` picks CUDA, then Apple Metal (MPS), then CPU. Measured on an M5 Pro
(20-core GPU): 3.3 img/s on CPU, 17 img/s on MPS.

`use_amp` is **off** in both trainers. fp16 overflows the unnormalised FPN-sum decoder,
and the inf poisons the head BatchNorms' running stats, which GradScaler does not cover
— the run then looks fine in `train()` and returns nan only in `eval()`. See the comment
on `use_amp` in trainer.py and DATASET.md §15.2. `assert_finite()` runs every epoch and
catches it within one epoch if you turn it back on.

## How to use
* `python trainer.py` to train a model.
* `torchrun --nproc_per_node=2 --nnodes=1 trainer_ddp.py` for multi-gpu.
* `python inference.py` to draw predictions for one random validation image.
* `python torch2onnx.py` converts the traced torch model to onnx.
* `python onnx2trt.py` converts the onnx model to a tensorRT engine.
* `python test_pipeline.py` checks box encode/decode, topk decode, augmentation box
  tracking, the rare-class split rule, mAP, the confusion matrix and the RoI head.
  `test_ground_truth_targets_score_perfect_map` is the one to run first if a training
  run reports mAP 0 — it tells a broken pipeline apart from an undertrained model.
* `python smoke_train.py` overfits 32 images end to end before you commit to a full
  run. Its mAP is tiny by design (~1/200 of a real run's steps); it checks that the
  chain is connected, not that the model is good. See DATASET.md §15.
* `python analyze_material_cues.py` measures which visual cue actually separates the
  9 confusable container classes (DATASET.md §13).
* `python analyze_stage2_ceiling.py` fine-tunes a classifier on ground-truth crops at
  several detail levels — the ceiling for any detect-then-classify cascade (§14).
* `python analyze_dataset.py` regenerates every number in docs/DATASET.md. Re-run it
  whenever the dataset changes — most of that document's conclusions are tied to
  specific values.
