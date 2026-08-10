import math

import numpy as np
import torch
from pt_dataset.dataset import ImgDataset


def collate_fn(batch):
    """Stack the fixed-shape targets, keep the per-image boxes as a list.

    Box count varies per image, so they cannot be stacked into one tensor.
    torchvision's roi_align takes exactly this list-of-(Ni,4) form.
    """
    images, hms, whs, offsets, masks, boxes, labels = zip(*batch)
    stacked = [torch.from_numpy(np.stack(a)) for a in (images, hms, whs, offsets, masks)]
    return (*stacked,
            [torch.from_numpy(b) for b in boxes],
            [torch.from_numpy(l) for l in labels])


def repeat_factors(annotation, thresh=0.05, cap=4.0):
    """
    LVIS-style repeat-factor sampling weights, one per image.

    r_c = clamp(sqrt(thresh / f_c), 1, cap) where f_c is the fraction of images
    containing class c; an image's weight is the max over the classes it contains.

    The metric averages AP over classes with equal weight while the data is 470:1
    imbalanced, so the rarest classes are the cheapest mAP available. This is unusually
    cheap here because rare-class images also contain common ones: measured on the real
    train split at thresh=0.05, the worst-off class loses 4% of its exposure while the
    rarest gains 5.6x, and anthropogenic_fragment - the worst-performing class - is not
    hurt at all. cap keeps a 13-box class from being memorised.
    """
    n = len(annotation)
    img_count = {}
    for item in annotation:
        for c in set(item["labels"]):
            img_count[c] = img_count.get(c, 0) + 1

    r_c = {c: min(cap, max(1.0, math.sqrt(thresh / (cnt / n))))
           for c, cnt in img_count.items()}
    return [max((r_c[c] for c in set(item["labels"])), default=1.0) for item in annotation]


def dataloader(train, valid, num_classes, batch_size, image_size, num_workers=8, cache=False,
               rfs_thresh=0.05, rfs_cap=4.0, mosaic_p=0.2, cutout_p=0.2,
               gauss_min_overlap=0.7):
    weights = repeat_factors(train, rfs_thresh, rfs_cap) if rfs_thresh else None

    train_dataset = ImgDataset(annotation=train, input_shape=image_size, num_classes=num_classes,
                               is_train=True, cache=cache, sample_weights=weights,
                               mosaic_p=mosaic_p, cutout_p=cutout_p,
                               gauss_min_overlap=gauss_min_overlap)
    # Validation targets use the same encoding as training, so the loss stays comparable.
    valid_dataset = ImgDataset(annotation=valid, input_shape=image_size, num_classes=num_classes,
                               is_train=False, cache=cache,
                               gauss_min_overlap=gauss_min_overlap)

    # num_samples is len(dataset), NOT sum(weights): the epoch stays exactly 12397 draws
    # and therefore 387 steps, so total_iters = epochs * len(loader) still describes the
    # cosine correctly. Repeating the dataset instead would stretch the epoch by 1.23x
    # and silently desynchronise the schedule - the failure this whole run is fixing.
    sampler = None
    if weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(train_dataset), replacement=True)
        print(f"RFS sampler: mean repeat {np.mean(weights):.3f}, max {max(weights):.2f}")

    print("Number of Train Data : ", train_dataset.get_number_data())
    print("Number of Valid Data : ", valid_dataset.get_number_data())

    # Pinned memory only speeds up a host->CUDA copy. On MPS the memory is already
    # unified and torch warns that the flag does nothing.
    pin_memory = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=(sampler is None),
                                                sampler=sampler,
                                                num_workers=num_workers,
                                                drop_last=True,
                                                persistent_workers=num_workers > 0,
                                                collate_fn=collate_fn,
                                                pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                drop_last=False,
                                                persistent_workers=num_workers > 0,
                                                collate_fn=collate_fn,
                                                pin_memory=pin_memory)

    return train_loader, valid_loader
