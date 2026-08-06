import torch
import torch_optimizer as optim_alg
from utils.tool import load_annotation, stratified_split, folderCheck
from utils.pytorchtools import (EarlyStopping, CosineDecayWarmup, traced_func, make_scaler,
                                assert_finite)
from pt_dataset.dataloader_ddp import dataloader
from model.centerNet import CenterNet
from model.loss import TotalLoss
from flow.flow import train, evaluate

import os
local_rank = int(os.environ["LOCAL_RANK"])

DEVICE = torch.device("cuda", local_rank)

torch.backends.cudnn.benchmark = True
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend='nccl')

torch.backends.cudnn.benchmark = True

batch_size = 16
input_shape = (512, 512)
epochs = 150
model_path = "savemodel"
use_amp = False         # see the note in trainer.py: fp16 poisons BatchNorm running stats

annotation_path = "data/train_dataset/train_label.json"
valid_ratio = 0.2

annotations, class_names = load_annotation(annotation_path)
num_classes = len(class_names)

# Same seed on every rank, so all ranks get an identical split. Only rank 0 writes it out.
train_annotation, valid_annotation = stratified_split(annotations, valid_ratio=valid_ratio,
                                                      save=(local_rank == 0))
train_loader, valid_loader, train_sampler, valid_sampler = dataloader(train=train_annotation,
                                                                      valid=valid_annotation,
                                                                      num_classes=num_classes,
                                                                      batch_size=batch_size,
                                                                      image_size=input_shape)
    
criterion = TotalLoss().to(DEVICE)
    
model = CenterNet(num_classes=num_classes).to(DEVICE)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

scaler = make_scaler(DEVICE, enabled=use_amp)

def all_reduce_mean(value, device):
    """Average a python scalar across ranks so every rank makes the same decision."""
    t = torch.tensor(value, device=device, dtype=torch.float32)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return (t / torch.distributed.get_world_size()).item()

optimizer = optim_alg.Ranger(model.parameters(), lr=1e-3, weight_decay=1e-3)

early_stopping = EarlyStopping(patience=30, verbose=False)

scheduler = CosineDecayWarmup(optimizer=optimizer, 
                            lr=1e-3, 
                            warmup_len=int(epochs*0.1) * len(train_loader), 
                            total_iters=epochs * len(train_loader))
    
if __name__ == "__main__":
    # torchrun --nproc_per_node=2 --nnodes=1 trainer_ddp.py
    best = 0
    train_losses = []
    valid_losses = []
    if local_rank == 0:
        folderCheck([model_path, "eval_fig"])

    for e in range(epochs):
        train_sampler.set_epoch(e)

        b_train_loss = train(now_ep=e,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            dataloader=train_loader,
                            criterion=criterion,
                            DEVICE=DEVICE,
                            scaler=scaler)

        assert_finite(model)
        b_valid_loss, map1, map2, _ = evaluate(mode="valid",
                                            model=model,
                                            dataloader=valid_loader,
                                            criterion=criterion,
                                            DEVICE=DEVICE,
                                            image_size=input_shape,
                                            amp=scaler.is_enabled(),
                                            num_classes=num_classes)

        # Each rank only sees its own shard. Average first, otherwise rank 0 would
        # checkpoint on 1/N of the validation set.
        # ponytail: averaging per-rank mAP is not the same as mAP over the whole set,
        # because AP is not linear in the samples. Close enough for checkpoint
        # selection; gather the raw detections if you need the exact number.
        b_train_loss = all_reduce_mean(b_train_loss, DEVICE)
        b_valid_loss = all_reduce_mean(b_valid_loss, DEVICE)
        map1 = all_reduce_mean(map1, DEVICE)
        map2 = all_reduce_mean(map2, DEVICE)

        if local_rank == 0:
            print(f"epoch {e}: loss {b_valid_loss:.3f}  mAP@0.5 stage1 {map1:.4f} -> reranked {map2:.4f}")
        train_losses.append(b_train_loss)
        valid_losses.append(b_valid_loss)
        # Every rank runs this on identical numbers, so they all break on the same epoch.
        early_stopping(b_valid_loss)

        if map2 >= best:
            best = map2
            if local_rank == 0:
                torch.save(model.module.state_dict(), model_path+'/model.pth')
                input_x = torch.rand(1, 3, input_shape[1], input_shape[0]).to(DEVICE)
                traced_model = traced_func(model.module, saved_path=model_path+'/model_trace.pt', X=input_x)

        if early_stopping.early_stop:
            if local_rank == 0:
                print("Early Stopping !! ")
            break

    torch.distributed.destroy_process_group()