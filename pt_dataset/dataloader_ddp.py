import torch
from torch.utils.data.distributed import DistributedSampler
from pt_dataset.dataset import ImgDataset

def dataloader(train, valid, num_classes, batch_size, image_size):
    train_dataset = ImgDataset(annotation=train, input_shape=image_size, num_classes=num_classes, is_train=True)
    valid_dataset = ImgDataset(annotation=valid, input_shape=image_size, num_classes=num_classes, is_train=False)

    print("Number of Train Data : ", train_dataset.get_number_data())
    print("Number of Valid Data : ", valid_dataset.get_number_data())

    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                sampler=train_sampler,
                                                num_workers=8,
                                                drop_last=True,
                                                persistent_workers=True,
                                                pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=batch_size,
                                                sampler=valid_sampler,
                                                num_workers=8,
                                                drop_last=False,
                                                persistent_workers=True,
                                                pin_memory=True)

    return train_loader, valid_loader, train_sampler, valid_sampler