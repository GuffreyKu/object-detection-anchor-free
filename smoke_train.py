"""Small-data end-to-end check before committing to a full training run.

    python smoke_train.py

Overfits a handful of images for a few epochs. Loss going down proves very little on
its own - the real check is that mAP RISES, because that only happens if the dataset,
the loss, the topk decode, the RoI head and the metric all agree with each other. A
falling loss with a flat mAP means something in that chain is disconnected.
"""
import torch


from utils.tool import load_annotation, stratified_split
from utils.pytorchtools import (get_device, make_scaler, CosineDecayWarmup, traced_func,
                                assert_finite)
from pt_dataset.dataset import ImgDataset
from pt_dataset.dataloader import collate_fn
from model.centerNet import CenterNet
from model.loss import TotalLoss
from flow.flow import train, evaluate

annotation_path = "data/train_dataset/train_label.json"
input_shape = (512, 512)
n_images, batch_size, epochs, lr, eval_every = 32, 8, 150, 1e-3, 25

# Adam, not the Ranger that trainer.py uses. Measured on a single synthetic box, after
# 300 steps Adam gets the heatmap peak to 0.88 (focal 0.67) while Ranger reaches 0.65
# (focal 20.5). Ranger may still win over a 150-epoch run, but this check is about
# whether the code is wired up, and waiting ~10x longer to find that out is wasteful.

if __name__ == "__main__":
    DEVICE = get_device()
    annotations, class_names = load_annotation(annotation_path)
    num_classes = len(class_names)
    tr, _ = stratified_split(annotations, valid_ratio=0.2, save=False)
    tr = tr[:n_images]

    # Augmentation off and the same images on both sides on purpose: this asks whether
    # the model CAN fit, not whether it generalises. If it cannot memorise 16 images,
    # nothing about a 12k-image run will be interpretable.
    dataset = ImgDataset(annotation=tr, input_shape=input_shape,
                         num_classes=num_classes, is_train=False)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn, persistent_workers=True)
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_fn, persistent_workers=True)

    model = CenterNet(num_classes=num_classes).to(DEVICE)
    criterion = TotalLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = make_scaler(DEVICE, enabled=False)   # matches trainer.py; see its use_amp note
    scheduler = CosineDecayWarmup(optimizer, lr=lr,
                                  warmup_len=len(train_loader),
                                  total_iters=epochs * len(train_loader))

    print(f"{num_classes} classes, {len(tr)} images (train == valid), amp={scaler.is_enabled()}")

    history = []
    for e in range(epochs):
        tl = train(e, model, optimizer, scheduler, train_loader, criterion, DEVICE, scaler=scaler)
        assert_finite(model)
        if e % eval_every and e != epochs - 1:
            continue
        vl, map1, map2, ev = evaluate("valid", model, valid_loader, criterion, DEVICE,
                                      image_size=input_shape, amp=scaler.is_enabled(),
                                      num_classes=num_classes)
        print(f"epoch {e:3}  train {tl:8.3f}  valid {vl:8.3f}  "
              f"mAP stage1 {map1:.4f}  reranked {map2:.4f}", flush=True)
        history.append((tl, map1, map2))

    print("\n" + ev.report(class_names))

    first_loss, last_loss = history[0][0], history[-1][0]
    best1 = max(h[1] for h in history)
    best2 = max(h[2] for h in history)

    # Do not raise this bar. ~600 steps is roughly 1/200 of a real run, and the focal
    # loss spends its first few hundred steps suppressing background (3 boxes against
    # 34*128*128 = 557k cells) before it starts lifting the centers at all. Measured
    # here: mAP is still 0 at step 300, 0.0003 at 404, 0.0043 at 504 and climbing. The
    # claim being checked is "the chain is connected and moving in the right
    # direction", not "the model is any good". Absolute quality needs the full run.
    assert last_loss < first_loss / 5, (f"loss barely moved: {first_loss:.1f} -> "
                                        f"{last_loss:.1f}; training is not working")
    assert best1 > 0, (f"mAP never left zero while loss fell {first_loss:.1f} -> "
                       f"{last_loss:.1f}. Run test_pipeline.py: if "
                       f"test_ground_truth_targets_score_perfect_map fails the eval path "
                       f"is broken, if it passes the model simply is not localising yet")
    print(f"\nloss {first_loss:.2f} -> {last_loss:.2f}, best mAP stage1 {best1:.4f} / "
          f"reranked {best2:.4f}  (tiny by design, see the note above)")

    model.eval()
    x = torch.rand(1, 3, input_shape[1], input_shape[0]).to(DEVICE)
    tm = traced_func(model, saved_path="/tmp/smoke_trace.pt", X=x)
    with torch.no_grad():
        assert all(torch.allclose(a, b, atol=1e-4) for a, b in zip(model(x), tm(x))), \
            "traced model diverged from eager"
    print("jit.trace still matches eager (stage 1 only, as expected)")
    print("SMOKE OK")
