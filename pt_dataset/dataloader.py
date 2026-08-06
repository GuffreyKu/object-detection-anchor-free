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


def dataloader(train, valid, num_classes, batch_size, image_size, num_workers=8):
    train_dataset = ImgDataset(annotation=train, input_shape=image_size, num_classes=num_classes, is_train=True)
    valid_dataset = ImgDataset(annotation=valid, input_shape=image_size, num_classes=num_classes, is_train=False)

    print("Number of Train Data : ", train_dataset.get_number_data())
    print("Number of Valid Data : ", valid_dataset.get_number_data())

    # Pinned memory only speeds up a host->CUDA copy. On MPS the memory is already
    # unified and torch warns that the flag does nothing.
    pin_memory = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
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
