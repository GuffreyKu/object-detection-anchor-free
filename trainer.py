import os
import random
import time

import numpy as np
import torch
import torch_optimizer as optim_alg
from utils.tool import load_annotation, stratified_split, folderCheck
from utils.pytorchtools import (EarlyStopping, CosineDecayWarmup, traced_func, get_device,
                                make_scaler, assert_finite, save_checkpoint, load_checkpoint,
                                checkpoint_guard)
from pt_dataset.dataloader import dataloader
from model.centerNet import CenterNet
from model.loss import TotalLoss, inverse_sqrt_class_weights
from flow.flow import train, evaluate

DEVICE = get_device()

# Fixed so two runs are comparable. Without this an A/B between recipes measures the
# seed as much as the change.
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

batch_size = 32
input_shape = (512, 512)
model_path = "savemodel"

# Schedule. `epochs` sizes the cosine, so it has to match how long the run actually gets
# to live - run 1 set 300, was killed by early stopping at 53, and therefore died at 18%
# of the schedule with the LR still at 98% of peak, having never annealed at all.
#   240 epochs x 630 s/epoch (455 s train + ~175 s eval) = 42 h, inside a 48 h budget.
# deadline_hours is the backstop: if epochs turn out slower than that, the cosine is
# reshaped to finish on time rather than being truncated mid-curve.
epochs = 240
warmup_epochs = 3           # was int(epochs*0.1) = 30, i.e. 26% of run 1 spent ramping
peak_lr = 1e-3              # unchanged: 0.4474 was produced at this LR, and the evidence
                            # from run 1 is underfitting, not too-high LR
min_lr = 1e-5
deadline_hours = 45.0
close_augment_at = 0.75     # fraction of the run after which mosaic and cutout stop
# Mixed precision. The autocast dtype is chosen by amp_dtype(): bf16 on any GPU that
# has it, fp16 otherwise.
#
# This used to be off, because in fp16 it produced nan. CenterNetDecoder is Conv2d +
# Upsample summing four FPN levels with no normalisation anywhere, so nothing bounds its
# activations and in fp16 they eventually pass 65504. The inf then lands in the head
# BatchNorms' running_mean / running_var, which are updated in the forward pass and so
# are NOT covered by GradScaler: parameters stay finite, the buffers are poisoned
# permanently, and train() keeps working because it uses batch statistics - only eval()
# shows it. Casting the heads to fp32 only delayed it (first failure moved from step 29
# to step 225), because the inf arrives already formed.
#
# bf16 fixes it at the source: same 8-bit exponent as fp32, so a decoder sum that
# overflowed fp16 at 65504 is just an ordinary number, and the ~34% speedup is no longer
# paid for with a lost run. It is not a substitute for normalising the decoder - the
# activations are still unbounded, they simply have ~1e33 more headroom now - so
# assert_finite() below stays, and on fp16-only hardware this should go back to False.
use_amp = True

annotation_path = "data/train_dataset/train_label.json"
valid_ratio = 0.2

# Decode the whole dataset into RAM once at startup instead of reading it per batch.
# Costs 11.9 GB resident (12397 + 2730 images at 512x512x3 uint8) and about a minute of
# startup. It does NOT make training faster here: measured at batch 32, a full step is
# 2237 ms and the dataloader accounts for 1 ms of that, because 8 workers keep the
# prefetch queue full behind a GPU-bound step. Worth it if the images live on slow or
# networked storage, or to stop hammering the disk; otherwise leave it off.
cache_images = True

# Feature extractor. Any name from model.backnone.BACKBONES; swin_t is the default.
# Backbone weights are not transferable, so switching this invalidates every checkpoint
# under savemodel/ - and both of them are overwritten by the first improving epoch, so
# archive them before changing this. checkpoint_guard() below refuses to start rather
# than let that happen silently.
backbone = "swin_t"

# Recipe knobs, grouped so an A/B can change one line each.
mosaic_p = 0.2              # was a hardcoded 0.5; see ImgDataset for why 0.2 not 0.25
cutout_p = 0.2              # was 0.5, rolled once per mosaic tile
gauss_min_overlap = 0.7     # CenterNet reference; was 0.3, which gave radius up to 63
rfs_thresh = 0.05           # repeat-factor sampling; 0 disables it
class_weighted_ce = True    # inverse-sqrt class weights on the second-stage CE

# Resume. last.pth is the full checkpoint - weights, optimizer momentum, position on the
# LR curve, early-stopping counter - rewritten every epoch. A missing file just starts a
# fresh run, so this can be left on.
#   None                  force a fresh run
#   savemodel/model.pth   warm-start from the best-mAP weights ALONE. That file is a bare
#                         state_dict, so the optimizer and the cosine schedule restart
#                         from zero; use it to seed a new run, not to continue one.
# The train/valid split is seeded (stratified_split, seed=42), so a resumed run keeps the
# same validation images and its mAP stays comparable to the run it continues.
resume = model_path + "/last.pth"

if __name__ == "__main__":
    folderCheck([model_path, "eval_fig"])
    annotations, class_names = load_annotation(annotation_path)
    num_classes = len(class_names)

    train_annotation, valid_annotation = stratified_split(annotations, valid_ratio=valid_ratio)
    print(f"{num_classes} classes, {len(train_annotation)} train / {len(valid_annotation)} valid images")

    train_loader, valid_loader = dataloader(train_annotation, valid_annotation, num_classes,
                                            batch_size, input_shape, cache=cache_images,
                                            rfs_thresh=rfs_thresh, mosaic_p=mosaic_p,
                                            cutout_p=cutout_p,
                                            gauss_min_overlap=gauss_min_overlap)
    
    box_counts = np.bincount(
        [c for a in train_annotation for c in a["labels"]], minlength=num_classes)
    criterion = TotalLoss(class_weights=inverse_sqrt_class_weights(box_counts)
                          if class_weighted_ce else None).to(DEVICE)

    checkpoint_guard([model_path+'/model.pth', model_path+'/last.pth'], backbone)
    model = CenterNet(num_classes=num_classes, backbone=backbone).to(DEVICE)

    optimizer = optim_alg.Ranger(model.parameters(), lr=peak_lr, weight_decay=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scaler = make_scaler(DEVICE, enabled=use_amp)
    # Patience above `epochs` disables it. Run 1 was stopped by this on VALIDATION LOSS,
    # which bottomed at epoch 23, while the mAP it checkpoints on was still climbing at
    # epoch 50 - the loss is dominated by the two head classes and the unweighted CE,
    # mAP is a uniform 1/34 average, and on a cosine you want the schedule to finish.
    # It now watches -map1, so if it ever does fire it fires on the right quantity.
    early_stopping = EarlyStopping(patience=epochs + 1, verbose=False)

    scheduler = CosineDecayWarmup(optimizer=optimizer,
                              lr=peak_lr,
                              warmup_len=warmup_epochs * len(train_loader),
                              total_iters=epochs * len(train_loader),
                              min_lr=min_lr)

    train_losses = []
    valid_losses = []
    map_history = []
    best = 0
    start_epoch = 0

    if resume and os.path.exists(resume):
        start_epoch, best, train_losses, valid_losses, map_history = load_checkpoint(
            resume, model, optimizer, scheduler, scaler, early_stopping,
            map_location=DEVICE)
    elif resume:
        print(f"resume: {resume} not found, starting from scratch")

    if start_epoch >= epochs:
        print(f"checkpoint is already at epoch {start_epoch} of {epochs}; "
              f"raise `epochs` to train further")

    started = time.time()
    e = start_epoch - 1
    for e in range(start_epoch, epochs):
        if e >= int(epochs * close_augment_at) and train_loader.dataset.mosaic_p:
            print(f"epoch {e}: closing mosaic and cutout for the anneal")
            train_loader.dataset.close_augment()

        b_train_loss = train(now_ep=e,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            dataloader=train_loader,
                            criterion=criterion,
                            DEVICE=DEVICE,
                            scaler=scaler)

        assert_finite(model)
        b_valid_loss, map1, map2, ev = evaluate(mode="valid",
                                            model=model,
                                            dataloader=valid_loader,
                                            criterion=criterion,
                                            DEVICE=DEVICE,
                                            image_size=input_shape,
                                            amp=scaler.is_enabled(),
                                            num_classes=num_classes)

        print(f"epoch {e}: loss {b_valid_loss:.3f}  mAP@0.5 {map1:.4f}  "
              f"lr {scheduler.get_lr():.2e}"
              + (f"  reranked {map2:.4f}" if map2 is not None else ""))
        train_losses.append(b_train_loss)
        valid_losses.append(b_valid_loss)
        map_history.append(map1)
        early_stopping(-map1)

        # Selection is on stage-1 mAP. Measured on the run-1 weights, the RoI rerank was
        # a 4.5-point regression and this line used to select on it.
        if map1 >= best:
            best = map1
            print(ev.report(class_names))
            torch.save(model.state_dict(), model_path+'/model.pth')

        # Every epoch, not only on improvement: this is the file a resume reads, so it
        # has to describe where the run actually is, not where it last did well.
        save_checkpoint(model_path+'/last.pth', e, model, optimizer, scheduler, scaler,
                        early_stopping, best, train_losses, valid_losses,
                        map_history=map_history)

        if e % 20 == 0:
            rare = [(class_names[c], len(np.concatenate(ev.scores[c])) if ev.scores[c] else 0)
                    for c in range(num_classes) if ev.n_gt[c] == 0]
            print(f"  classes with no validation GT, prediction counts: {rare}")

        # The cosine is sized in epochs, so if the run is slower than planned the anneal
        # has to be pulled in rather than truncated - that truncation is exactly what
        # cost run 1. range() was already evaluated, so reassigning `epochs` does nothing
        # to the loop; the explicit break below is what actually ends it.
        spent = time.time() - started
        per_epoch = spent / (e - start_epoch + 1)
        affordable = int((deadline_hours * 3600 - spent) / per_epoch)
        if affordable < epochs - e - 1:
            epochs = e + 1 + max(1, affordable)
            scheduler.total_iters = epochs * len(train_loader)
            print(f"  deadline: {per_epoch/60:.1f} min/epoch, reshaping cosine to end at "
                  f"epoch {epochs - 1}")
        if e + 1 >= epochs:
            break

        if early_stopping.early_stop:
            print("Early Stopping !! ")
            break

    # Traced once at the end, on the best weights, instead of on every improvement -
    # that wrote 123 MB and re-traced on nearly every early epoch.
    print(f"\nbest stage-1 mAP@0.5 {best:.4f} over {e - start_epoch + 1} epochs")
    model.load_state_dict(torch.load(model_path+'/model.pth', map_location=DEVICE))
    input_x = torch.rand(1, 3, input_shape[1], input_shape[0]).to(DEVICE)
    traced_func(model, saved_path=model_path+'/model_trace.pt', X=input_x)