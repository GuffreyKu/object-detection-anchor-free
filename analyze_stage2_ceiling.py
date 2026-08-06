"""Ceiling for a two-stage design: how well can a classifier do on GROUND TRUTH crops?

    python analyze_stage2_ceiling.py

Stage 2 of a detect-then-classify cascade can never beat a classifier fed perfect
boxes. This measures that ceiling on the 9 confusable container classes (DATASET.md
7.3), and separately asks whether crop RESOLUTION is where the benefit comes from:
crops are cached at 224 from the ORIGINAL image, then degraded to N px and upsampled
back to 224, so the network input size is constant and only information differs.
64px is roughly what a median box occupies inside the 512 detector input.
"""
import json, random, collections
import cv2, numpy as np, torch, torch.nn as nn

from pt_dataset.dataUtils import read_image_rgb
import torchvision

from utils.pytorchtools import get_device

GROUP = ["plastic_bottle", "non_food_plastic_bottle", "glass_bottle", "metal_can",
         "non_pet_food_beverage_container", "drink_carton", "non_pet_food_container",
         "non_food_plastic_container", "aluminum_packaging"]
PER_CLASS, R = 500, 224
DEV = get_device()

d = json.load(open("data/train_dataset/train_label.json"))
imgs = {i["id"]: i for i in d["images"]}
name = {c["id"]: c["name"] for c in d["categories"]}
by_cat = collections.defaultdict(list)
for a in d["annotations"]:
    if name[a["category_id"]] in GROUP and min(a["bbox"][2], a["bbox"][3]) > 24:
        by_cat[name[a["category_id"]]].append(a)

rng = random.Random(0)
picked = []
for c in GROUP:
    anns = by_cat[c][:]
    rng.shuffle(anns)
    picked += [(a, GROUP.index(c)) for a in anns[:PER_CLASS]]
print("crops:", {c: min(len(by_cat[c]), PER_CLASS) for c in GROUP})

per_img = collections.defaultdict(list)
for a, y in picked:
    per_img[a["image_id"]].append((a, y))

X, Y, IMG = [], [], []
for k, (img_id, items) in enumerate(per_img.items()):
    # same frame the annotations use, see pt_dataset.dataUtils.read_image_rgb
    im = read_image_rgb("data/train_dataset/images/" + imgs[img_id]["filename"])
    H, W = im.shape[:2]
    for a, y in items:
        x, yy, w, h = a["bbox"]
        x1, y1, x2, y2 = max(0, int(x)), max(0, int(yy)), min(W, int(x + w)), min(H, int(yy + h))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        X.append(cv2.resize(im[y1:y2, x1:x2], (R, R), interpolation=cv2.INTER_AREA))
        Y.append(y); IMG.append(img_id)
    if k % 400 == 0:
        print(f"  {k}/{len(per_img)}", flush=True)

X, Y, IMG = np.array(X), np.array(Y), np.array(IMG)
print("total crops:", len(Y), " class counts:", np.bincount(Y).tolist())

# Split by IMAGE, not by crop: two crops from one photo share lighting and background,
# so a crop-level split would leak and inflate every number below.
uimg = sorted(set(IMG.tolist()))
random.Random(1).shuffle(uimg)
val_imgs = set(uimg[:len(uimg) // 4])
va = np.array([i in val_imgs for i in IMG])
print(f"{(~va).sum()} train / {va.sum()} valid crops from disjoint images")

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def degrade(batch, px):
    if px >= R:
        return batch
    small = torch.nn.functional.interpolate(batch, size=px, mode="area")
    return torch.nn.functional.interpolate(small, size=R, mode="bilinear", align_corners=False)


def run(px, epochs=12, seed=0):
    torch.manual_seed(seed)
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(512, len(GROUP))
    m = m.to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    tr_idx = np.where(~va)[0]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, 3e-4, total_steps=epochs * (len(tr_idx) // 64 + 1))
    for ep in range(epochs):
        m.train()
        for i in range(0, len(tr_idx), 64):
            b = tr_idx[i:i + 64]
            x = torch.from_numpy(X[b]).permute(0, 3, 1, 2).contiguous().float() / 255
            if random.random() < 0.5:
                x = torch.flip(x, [3]).contiguous()
            x = degrade(x, px)
            x = ((x - MEAN) / STD).to(DEV)
            loss = nn.functional.cross_entropy(m(x), torch.from_numpy(Y[b]).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        np.random.shuffle(tr_idx)
    m.eval()
    correct = tot = 0
    per_cls = collections.Counter(); per_cls_n = collections.Counter()
    with torch.no_grad():
        vi = np.where(va)[0]
        for i in range(0, len(vi), 64):
            b = vi[i:i + 64]
            x = torch.from_numpy(X[b]).permute(0, 3, 1, 2).contiguous().float() / 255
            x = ((degrade(x, px) - MEAN) / STD).to(DEV)
            p = m(x).argmax(1).cpu().numpy()
            correct += (p == Y[b]).sum(); tot += len(b)
            for t, q in zip(Y[b], p):
                per_cls_n[t] += 1; per_cls[t] += int(t == q)
    return correct / tot, {GROUP[c]: per_cls[c] / per_cls_n[c] for c in sorted(per_cls_n)}


print(f"\n{'crop detail':16}{'top-1':>9}{'vs chance':>11}   (chance {1/len(GROUP):.1%})")
res = {}
for px in (32, 64, 112, 224):
    a, pc = run(px)
    res[px] = pc
    tag = f"{px}px" + ("  <- ~detector" if px == 64 else "  <- full crop" if px == 224 else "")
    print(f"{tag:16}{a:>9.1%}{a/(1/len(GROUP)):>10.2f}x", flush=True)

print(f"\nper-class recall  {'64px':>8}{'224px':>8}")
for c in GROUP:
    if c in res[64]:
        print(f"{c:32}{res[64][c]:8.0%}{res[224][c]:8.0%}")
