"""
Regenerate every number in DATASET.md.

    python analyze_dataset.py [path/to/train_label.json]

Run this after any dataset change and update DATASET.md from the output.
"""
import sys
import json
import math
import collections
import numpy as np

from pt_dataset.dataUtils import gaussian_radius

INPUT_SIZE = 512


def load(path):
    with open(path) as f:
        d = json.load(f)
    imgs = {i["id"]: i for i in d["images"]}
    names = {c["id"]: c["name"] for c in d["categories"]}
    return d["annotations"], imgs, names


def section(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main(path):
    A, imgs, names = load(path)
    N = len(names)
    W = np.array([a["bbox"][2] for a in A], float)
    H = np.array([a["bbox"][3] for a in A], float)
    IW = np.array([imgs[a["image_id"]]["width"] for a in A], float)
    IH = np.array([imgs[a["image_id"]]["height"] for a in A], float)
    area = W * H
    rel = np.sqrt(area / (IW * IH))
    w512, h512 = W * INPUT_SIZE / IW, H * INPUT_SIZE / IH

    section(f"OVERALL  {len(A)} boxes / {len(imgs)} images / {N} classes")
    print(f"{'':16}{'p1':>9}{'p25':>9}{'median':>9}{'p75':>9}{'p99':>9}{'max':>9}")
    for label, x, f in [("width px", W, "{:9.0f}"), ("height px", H, "{:9.0f}"),
                        ("sqrt(area) px", np.sqrt(area), "{:9.0f}"),
                        ("rel size", rel, "{:9.3f}"), ("aspect w/h", W / H, "{:9.2f}")]:
        print(f"{label:16}" + "".join(f.format(np.percentile(x, p)) for p in (1, 25, 50, 75, 99, 100)))

    small = int((area < 32 ** 2).sum())
    med = int(((area >= 32 ** 2) & (area < 96 ** 2)).sum())
    large = int((area >= 96 ** 2).sum())
    print(f"\nCOCO scale: small {small} ({small/len(A):.1%})  "
          f"medium {med} ({med/len(A):.1%})  large {large} ({large/len(A):.1%})")

    section("BOXES PER IMAGE")
    per = collections.Counter(a["image_id"] for a in A)
    pv = np.array(list(per.values()))
    dist = collections.Counter(pv.tolist())
    print(f"mean {pv.mean():.2f}  median {int(np.median(pv))}  "
          f"p95 {int(np.percentile(pv, 95))}  max {pv.max()}")
    cum = 0
    for k in range(1, 6):
        cum += dist.get(k, 0)
        print(f"  {k} box(es): {dist.get(k,0):6} images {dist.get(k,0)/len(pv):6.1%}  (cum {cum/len(pv):5.1%})")

    section(f"AFTER THE {INPUT_SIZE}x{INPUT_SIZE} RESIZE")
    for p in (1, 25, 50, 75, 99):
        print(f"  p{p:<3} w={np.percentile(w512,p):7.1f}  h={np.percentile(h512,p):7.1f}")
    for thr in (4, 8, 16):
        n = int(((w512 < thr) | (h512 < thr)).sum())
        print(f"  a side < {thr:2}px : {n:6} ({n/len(A):.2%})")
    r = IW / IH
    print(f"  aspect distortion from the stretch: up to {r.max()/r.min():.2f}x "
          f"(p1-p99 spread {np.percentile(r,99)/np.percentile(r,1):.2f}x)")

    section("STRIDE: CENTER COLLISIONS AND GAUSSIAN RADIUS")
    print("A collision means two boxes share a heatmap cell, so encode_targets silently drops one.")
    print(f"{'stride':>7}{'heatmap':>12}{'collisions':>12}{'pct':>9}{'radius med':>12}{'radius==0':>11}")
    for stride in (2, 4, 8):
        by = collections.defaultdict(list)
        for a in A:
            im = imgs[a["image_id"]]
            x, y, w, h = a["bbox"]
            by[a["image_id"]].append((int((x + w / 2) * INPUT_SIZE / im["width"] / stride),
                                      int((y + h / 2) * INPUT_SIZE / im["height"] / stride)))
        lost = 0
        for v in by.values():
            seen = set()
            for c in v:
                if c in seen:
                    lost += 1
                seen.add(c)
        rad = np.array([max(0, int(gaussian_radius((math.ceil(h / stride), math.ceil(w / stride)))))
                        for w, h in zip(w512, h512)])
        g = INPUT_SIZE // stride
        print(f"{stride:>7}{f'{g}x{g}':>12}{lost:>12}{lost/len(A):>9.3%}"
              f"{int(np.median(rad)):>12}{int((rad==0).sum()):>11}")

    section("CLASS DISTRIBUTION")
    by_cat = collections.defaultdict(list)
    for a in A:
        by_cat[a["category_id"]].append(a)
    rows = []
    for cid, anns in by_cat.items():
        w = np.array([x["bbox"][2] for x in anns], float)
        h = np.array([x["bbox"][3] for x in anns], float)
        iw = np.array([imgs[x["image_id"]]["width"] for x in anns], float)
        ih = np.array([imgs[x["image_id"]]["height"] for x in anns], float)
        rows.append((names[cid], len(anns), len({x["image_id"] for x in anns}),
                     float(np.median(np.sqrt(w * h / (iw * ih)))), float(np.median(w / h)),
                     int(np.median(w * INPUT_SIZE / iw)), int(np.median(h * INPUT_SIZE / ih))))
    rows.sort(key=lambda r: -r[1])
    print(f"{'class':32}{'boxes':>7}{'imgs':>7}{'rel':>7}{'w/h':>6}{'w@512':>7}{'h@512':>7}")
    for r in rows:
        print(f"{r[0][:31]:32}{r[1]:7}{r[2]:7}{r[3]:7.3f}{r[4]:6.2f}{r[5]:7}{r[6]:7}")

    section("mAP@0.5 CEILING  (AP is averaged per class, so each class weighs 1/N)")
    tail = rows[-14:]
    tb = sum(r[1] for r in tail)
    print(f"the {len(tail)} smallest classes hold {tb} boxes "
          f"({tb/len(A):.1%} of boxes) but {len(tail)/N:.1%} of the mAP weight")
    print(f"\n{'classes written off':>21}{'remaining':>11}{'avg AP needed for 0.60':>26}")
    for K in (0, 5, 8, 10, 12, 14):
        need = 0.60 * N / (N - K)
        note = "  impossible" if need > 1 else ("  very hard" if need > 0.85 else "")
        print(f"{K:>21}{N-K:>11}{need:>26.3f}{note}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/train_dataset/train_label.json")
