"""Which visual cue actually separates the bottle / can / glass group?

    python analyze_material_cues.py

Crops every GT box of the 9 confusable container classes (DATASET.md 7.3), extracts
several families of hand-crafted features, and fits a linear classifier on each.
Held-out accuracy against 1/9 = 11.1% chance says what the signal is made of.

These are global statistics over a 96x96 crop read by a LINEAR model. The absolute
numbers are a floor on what a CNN can do, not a ceiling. Only the ranking between
feature families is the point.
"""
import json, random, collections
import cv2, numpy as np, torch

from pt_dataset.dataUtils import read_image_rgb

GROUP = ["plastic_bottle", "non_food_plastic_bottle", "glass_bottle", "metal_can",
         "non_pet_food_beverage_container", "drink_carton", "non_pet_food_container",
         "non_food_plastic_container", "aluminum_packaging"]
PER_CLASS = 220
CROP = 96

d = json.load(open("data/train_dataset/train_label.json"))
imgs = {i["id"]: i for i in d["images"]}
name = {c["id"]: c["name"] for c in d["categories"]}
cid = {c["name"]: c["id"] for c in d["categories"]}

by_cat = collections.defaultdict(list)
for a in d["annotations"]:
    if name[a["category_id"]] in GROUP and min(a["bbox"][2], a["bbox"][3]) > 24:
        by_cat[a["category_id"]].append(a)

rng = random.Random(0)
picked = []
for c in GROUP:
    anns = by_cat[cid[c]]
    rng.shuffle(anns)
    picked += [(a, GROUP.index(c)) for a in anns[:PER_CLASS]]
print("crops:", collections.Counter(GROUP[y] for _, y in picked))

# group crops by image so each file is decoded once
per_img = collections.defaultdict(list)
for a, y in picked:
    per_img[a["image_id"]].append((a, y))


def feats(c):
    """c: uint8 RGB crop, already resized to CROPxCROP."""
    f = {}
    rgb = c.astype(np.float32) / 255
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[..., 0] * np.pi / 90.          # 0..179 -> radians
    s, v = hsv[..., 1] / 255, hsv[..., 2] / 255
    lab = cv2.cvtColor(c, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)

    # --- hue only: circular mean and concentration, weighted by saturation ---
    w = s / (s.sum() + 1e-6)
    cx, cy = (np.cos(h) * w).sum(), (np.sin(h) * w).sum()
    f["hue"] = [cx, cy, np.hypot(cx, cy)]

    # --- plain colour ---
    f["rgb"] = list(rgb.reshape(-1, 3).mean(0)) + list(rgb.reshape(-1, 3).std(0))
    f["colour"] = (f["rgb"] + [s.mean(), s.std(), v.mean(), v.std()]
                   + list(lab.reshape(-1, 3).mean(0) / 255) + list(lab.reshape(-1, 3).std(0) / 255))

    # --- specular / dichromatic: how bright pixels differ from the body ---
    hi = v >= np.quantile(v, 0.95)
    lo = v <= np.quantile(v, 0.50)
    f["specular"] = [
        float(hi.mean() and s[hi].mean()),        # saturation of the highlight
        float(s[lo].mean()),                      # saturation of the body
        float(s[lo].mean() - (hi.mean() and s[hi].mean())),  # dielectric separation
        float(v.max()), float(np.quantile(v, 0.99) - np.quantile(v, 0.5)),
        float((v > 0.97).mean()),                 # clipped-white area, metal blows out
    ]

    # --- texture / micro-contrast ---
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    f["texture"] = [float(lap.var()) / 1e3, float(mag.mean()) / 100, float(mag.std()) / 100,
                    float(gray.std()) / 100, float(np.quantile(mag, 0.9)) / 100]
    return f


X, Y = collections.defaultdict(list), []
for k, (img_id, items) in enumerate(per_img.items()):
    # same frame the annotations use, see pt_dataset.dataUtils.read_image_rgb
    im = read_image_rgb("data/train_dataset/images/" + imgs[img_id]["filename"])
    H, W = im.shape[:2]
    for a, y in items:
        x, yy, w, hh = a["bbox"]
        x1, y1 = max(0, int(x)), max(0, int(yy))
        x2, y2 = min(W, int(x + w)), min(H, int(yy + hh))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crop = cv2.resize(im[y1:y2, x1:x2], (CROP, CROP))
        for name_, vals in feats(crop).items():
            X[name_].append(vals)
        Y.append(y)
    if k % 300 == 0:
        print(f"  {k}/{len(per_img)} images", flush=True)

Y = np.array(Y)
print("total crops:", len(Y))

X["colour+specular"] = [a + b for a, b in zip(X["colour"], X["specular"])]
X["all"] = [a + b + c for a, b, c in zip(X["colour"], X["specular"], X["texture"])]


def acc(feat, y, seed=0):
    x = np.nan_to_num(np.array(feat, np.float32))
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g).numpy()
    x, y = x[perm], y[perm]
    n = int(len(y) * 0.75)
    xt, yt = torch.tensor(x[:n]), torch.tensor(y[:n])
    xv, yv = torch.tensor(x[n:]), torch.tensor(y[n:])
    m = torch.nn.Linear(x.shape[1], 9)
    opt = torch.optim.Adam(m.parameters(), lr=0.05, weight_decay=1e-3)
    for _ in range(600):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(xt), yt).backward()
        opt.step()
    return (m(xv).argmax(1) == yv).float().mean().item()


print(f"\n{'feature family':22}{'dims':>6}{'held-out acc':>14}{'vs chance':>11}")
print(f"{'(chance)':22}{'':>6}{1/9:>14.1%}")
for k in ["hue", "rgb", "colour", "specular", "texture", "colour+specular", "all"]:
    a = np.mean([acc(X[k], Y, s) for s in range(3)])
    print(f"{k:22}{len(X[k][0]):>6}{a:>14.1%}{a/(1/9):>10.2f}x")
