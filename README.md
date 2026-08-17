# object-detection-anchor-free

CenterNet-style anchor-free detector for marine litter, 34 classes.
Swin-T backbone (swappable), FPN-ish decoder, heatmap + wh + offset heads at stride 4,
RGB in.

See [docs/DATASET.md](docs/DATASET.md) for the distribution analysis and what it means
for modelling — class imbalance, stride choice, augmentation limits, and the known
blockers on mAP@0.5.

See [docs/STAGE2_EVAL.md](docs/STAGE2_EVAL.md) for `savemodel_two_stage`'s measured
mAP@0.5, per-class breakdown, and where the two-stage pipeline's errors actually come
from — proposal recall vs classifier confusion, and which class confusions are not
covered by the semantic groups in DATASET.md §7.3.

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

`use_amp` is **on** in both trainers, and `amp_dtype()` picks **bf16** on any GPU that
supports it. It used to be off: fp16 overflows the unnormalised FPN-sum decoder at 65504,
and the inf poisons the head BatchNorms' running stats, which GradScaler does not cover
— the run then looks fine in `train()` and returns nan only in `eval()`. bf16 has fp32's
exponent range, so that overflow cannot happen. On fp16-only hardware `amp_dtype()` falls
back to fp16 and `use_amp` should be set to `False`. See the comment on `use_amp` in
trainer.py and DATASET.md §15.2. `assert_finite()` still runs every epoch.

## Backbone
`backbone` in trainer.py takes any name from `model.backnone.BACKBONES`
(`resnet50`, `efficientnet_b3`, `mobilenet_v3_large`, `swin_t/s/b`, `swin_v2_t/s/b`), or
you can pass a ready `nn.Module` to `CenterNet(backbone=...)`. Every backbone returns
four feature maps at strides **4/8/16/32** and declares their widths in `out_channels`,
which is what `CenterNetDecoder` builds its lateral 1x1 convs from. Stride 4 at the
finest level is required: `pt_dataset/dataset.py` encodes targets at stride 4 and
`RoIClassifier` pools at `spatial_scale=1/4`, and a stride-8 backbone does not raise —
it just trains against targets at twice the intended scale. `python -m model.backnone`
checks the contract for every entry, as does a test in `test_pipeline.py`.

Measured here, batch 32 at 512x512 with bf16, forward+backward+step:

| backbone | params | ms/step | peak mem | ImageNet top-1 |
|---|---|---|---|---|
| resnet50 | 27.31 M | 790 | 10.6 GiB | 76.1 |
| **swin_t** (default) | 30.70 M | **1175** (1.49x) | **16.2 GiB** | 81.5 |

Swin rather than a plain ViT because its relative position bias is window-relative, so
the 224 pretrained weights transfer to 512 with no position-embedding surgery, and it
emits four resolutions natively. A torchvision `vit_*` hard-asserts `image_size == 224`,
fails `torch.jit.trace` at 512, and cannot export to ONNX (`aten::_native_multi_head_attention`).

Two Swin-specific notes. The backbone adds a `LayerNorm` per output stage, because
torchvision normalises only after the last stage and the raw taps peak at 10/10/298/158
against resnet50's 3/3/2/9 — the decoder has no normalisation anywhere, so those go
straight into the four-level sum (measured decoder output 241 without, 40 with, against
resnet50's 32). And `torch.jit.trace` now bakes in the window-padding decision, so
`model_trace.pt` is valid **only at 512x512**; everything in-repo already uses 512.

**Backbone weights are not transferable.** Switching `backbone` invalidates every
checkpoint under `savemodel/`, and `trainer.py` overwrites both files early in a run —
`checkpoint_guard()` refuses to start rather than let that happen silently.

## Image cache
By default images are read from disk per batch; 8 DataLoader workers keep the prefetch
queue full. `cache_images` in trainer.py decodes the whole dataset into RAM once at
startup instead — 11.9 GB resident, ~27 s to build, images stored already resized to
`input_shape` (the originals decode to 118 GB and do not fit).

It is not a speedup on this machine: measured at batch 32, a full training step is
2237 ms and the dataloader accounts for 1 ms of it, so the loader is entirely hidden
behind a GPU-bound step. Use it when the images live on slow or networked storage.
The cache is one contiguous array so the forked workers share it copy-on-write —
measured 12.6 GB resident, 17.6 GB with all 8 workers up, not 12 GB per process.

## Resuming
`trainer.py` writes two files into `savemodel/`, and they are not interchangeable:

* `model.pth` — the **best-mAP** snapshot, a bare `state_dict`, written only when the
  reranked mAP improves. This is what `inference.py` and `torch2onnx.py` load.
* `last.pth` — the **full checkpoint** for resuming, rewritten every epoch: weights,
  optimizer momentum, position on the cosine LR curve, early-stopping counter, best
  score and loss history.

Resume is controlled by `resume` in trainer.py and defaults to `savemodel/last.pth`; if
the file is not there the run just starts fresh. Set it to `None` to force a fresh run.
Pointing it at `model.pth` instead is a *warm start*, not a resume — that file has no
optimizer state or LR position in it, so both restart from zero. The train/valid split is
seeded, so a resumed run keeps the same validation images and comparable mAP.

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
