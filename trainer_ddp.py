import torch
import torch_optimizer as optim_alg
from utils.tool import load_annotation, train_test_split, folderCheck
from utils.pytorchtools import EarlyStopping, CosineDecayWarmup, traced_func
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
num_classes = 1

annotations_paths=["data/v2.json", "data/v3.json"]
annotations = []
for annotations_path in annotations_paths:
    annotations.extend(load_annotation(annotations_path))

train_annotation, valid_annotation = train_test_split(annotations)
train_loader, valid_loader, train_sampler, valid_sampler = dataloader(train=train_annotation, 
                                                                      valid=valid_annotation, 
                                                                      num_classes=num_classes, 
                                                                      batch_size=batch_size, 
                                                                      image_size=input_shape)
    
criterion = TotalLoss().to(DEVICE)
    
model = CenterNet(num_classes=num_classes).to(DEVICE)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)   

optimizer = optim_alg.Ranger(model.parameters(), lr=1e-3, weight_decay=1e-3)

early_stopping = EarlyStopping(patience=30, verbose=False)

scheduler = CosineDecayWarmup(optimizer=optimizer, 
                            lr=1e-3, 
                            warmup_len=int(epochs*0.1) * len(train_loader), 
                            total_iters=epochs * len(train_loader))
    
if __name__ == "__main__":
    # torchrun --nproc_per_node=2 --nnodes=1 trainer_ddp.py
    if local_rank == 0:
        folderCheck([model_path, "eval_fig"])
        best = 0
        train_losses = []
        valid_losses = []
        # torch.distributed.barrier()

    for e in range(epochs):
        train_sampler.set_epoch(e)
        valid_sampler.set_epoch(e)

        b_train_loss, b_train_iou = train(now_ep=e,
                                        model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        dataloader=train_loader,
                                        criterion=criterion,
                                        DEVICE=DEVICE)
        
        b_valid_loss, b_valid_iou = evaluate(mode="valid",
                                            model=model,
                                            dataloader=valid_loader,
                                            criterion=criterion,
                                            DEVICE=DEVICE)
        
        if local_rank == 0:
            train_losses.append(b_train_loss)
            valid_losses.append(b_valid_loss)
            early_stopping(b_valid_loss)

            if b_valid_iou >= best:
                best = b_valid_iou
                torch.save(model.module.state_dict(), model_path+'/model.pth')
                input_x = torch.rand(1, 1, input_shape[1], input_shape[0]).to(DEVICE)
                traced_model = traced_func(model.module, saved_path=model_path+'/model_trace.pt', X=input_x)
            
            # if early_stopping.early_stop:
            #     print("Early Stopping !! ")
            #     break