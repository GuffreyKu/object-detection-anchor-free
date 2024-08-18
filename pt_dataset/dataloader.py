import torch
from pt_dataset.dataset import ImgDataset

def dataloader(train, valid, num_classes, batch_size, image_size):
    train_dataset = ImgDataset(annotation=train, input_shape=image_size, num_classes=num_classes, is_train=True)
    valid_dataset = ImgDataset(annotation=valid, input_shape=image_size, num_classes=num_classes, is_train=False)

    print("Number of Train Data : ", train_dataset.get_number_data())
    print("Number of Valid Data : ", valid_dataset.get_number_data())


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=8,
                                                drop_last=True,
                                                persistent_workers=True,
                                                pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=8,
                                                drop_last=False,
                                                persistent_workers=True,
                                                pin_memory=True)

    return train_loader, valid_loader