import torch
import torch_optimizer as optim_alg
from utils.tool import load_annotation, stratified_split, folderCheck
from utils.pytorchtools import (EarlyStopping, CosineDecayWarmup, traced_func, get_device,
                                make_scaler, assert_finite)
from pt_dataset.dataloader import dataloader
from model.centerNet import CenterNet
from model.loss import TotalLoss
from flow.flow import train, evaluate

DEVICE = get_device()

batch_size = 32
input_shape = (512, 512)
epochs = 300
model_path = "savemodel"
# fp16 autocast, off by default. Measured on 32 images, 20 epochs, Adam lr 1e-3:
#   amp off -> loss 48.4, stable        amp on -> nan
#   amp off, lr 3e-4 -> loss 148.2      amp on, lr 3e-4 -> loss 166.9
#
# Why it breaks: CenterNetDecoder is Conv2d + Upsample summing four FPN levels with no
# normalisation anywhere, so nothing bounds its activations and in fp16 they eventually
# pass 65504. The inf then lands in the head BatchNorms' running_mean / running_var,
# which are updated in the forward pass and so are NOT covered by GradScaler: parameters
# stay finite, the buffers are poisoned permanently, and train() keeps working because it
# uses batch statistics - only eval() shows it. Casting the heads to fp32 delays this
# (first failure moved from step 29 to step 225) but cannot prevent it, because the inf
# arrives already formed. A real fix means normalising the decoder, which is an
# architecture change and needs its own validation.
#
# The ~34% speedup is not worth silently losing a run. assert_finite() below catches it
# within one epoch if you do turn this on.
use_amp = False

annotation_path = "data/train_dataset/train_label.json"
valid_ratio = 0.2

if __name__ == "__main__":
    folderCheck([model_path, "eval_fig"])
    annotations, class_names = load_annotation(annotation_path)
    num_classes = len(class_names)

    train_annotation, valid_annotation = stratified_split(annotations, valid_ratio=valid_ratio)
    print(f"{num_classes} classes, {len(train_annotation)} train / {len(valid_annotation)} valid images")

    train_loader, valid_loader = dataloader(train_annotation, valid_annotation, num_classes, batch_size, input_shape)
    
    criterion = TotalLoss().to(DEVICE)
    
    model = CenterNet(num_classes=num_classes).to(DEVICE)

    optimizer = optim_alg.Ranger(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scaler = make_scaler(DEVICE, enabled=use_amp)
    early_stopping = EarlyStopping(patience=30, verbose=False)

    scheduler = CosineDecayWarmup(optimizer=optimizer, 
                              lr=1e-3, 
                              warmup_len=int(epochs*0.1) * len(train_loader), 
                              total_iters=epochs * len(train_loader))
    
    train_losses = []
    valid_losses = []
    best = 0
    for e in range(epochs):
        b_train_loss = train(now_ep=e,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            dataloader=train_loader,
                            criterion=criterion,
                            DEVICE=DEVICE,
                            scaler=scaler)

        assert_finite(model)
        b_valid_loss, map1, map2, ev = evaluate(mode="valid",
                                            model=model,
                                            dataloader=valid_loader,
                                            criterion=criterion,
                                            DEVICE=DEVICE,
                                            image_size=input_shape,
                                            amp=scaler.is_enabled(),
                                            num_classes=num_classes)

        print(f"epoch {e}: loss {b_valid_loss:.3f}  mAP@0.5 stage1 {map1:.4f} -> reranked {map2:.4f}")
        train_losses.append(b_train_loss)
        valid_losses.append(b_valid_loss)
        early_stopping(b_valid_loss)

        if map2 >= best:
            best = map2
            print(ev.report(class_names))
            torch.save(model.state_dict(), model_path+'/model.pth')
            input_x = torch.rand(1, 3, input_shape[1], input_shape[0]).to(DEVICE)
            traced_model = traced_func(model, saved_path=model_path+'/model_trace.pt', X=input_x)
        
        if early_stopping.early_stop:
            print("Early Stopping !! ")
            break