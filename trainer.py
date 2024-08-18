import torch
import torch_optimizer as optim_alg
from utils.tool import load_annotation, train_test_split, folderCheck
from utils.pytorchtools import EarlyStopping, CosineDecayWarmup, traced_func
from pt_dataset.dataloader import dataloader
from model.centerNet import CenterNet
from model.loss import TotalLoss
from flow.flow import train, evaluate

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

batch_size = 16
input_shape = (512, 512)
epochs = 150
model_path = "savemodel"
num_classes = 3

annotations_paths=["data/v1.json"]
if __name__ == "__main__":
    folderCheck([model_path, "eval_fig"])
    annotations = []
    for annotations_path in annotations_paths:
        annotations.extend(load_annotation(annotations_path))

    train_annotation, valid_annotation = train_test_split(annotations)
    train_loader, valid_loader = dataloader(train_annotation, valid_annotation, num_classes, batch_size, input_shape)
    
    criterion = TotalLoss().to(DEVICE)
    
    model = CenterNet(num_classes=num_classes).to(DEVICE)
    # model = torch.load('savemodel/model.pt', map_location="cpu")
    model = model.to(DEVICE)

    optimizer = optim_alg.Ranger(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)

    early_stopping = EarlyStopping(patience=30, verbose=False)

    scheduler = CosineDecayWarmup(optimizer=optimizer, 
                              lr=1e-3, 
                              warmup_len=int(epochs*0.1) * len(train_loader), 
                              total_iters=epochs * len(train_loader))
    
    train_losses = []
    valid_losses = []
    best = 0
    for e in range(epochs):
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
        
        train_losses.append(b_train_loss)
        valid_losses.append(b_valid_loss)
        early_stopping(b_valid_loss)

        if b_valid_iou >= best:
            best = b_valid_iou
            torch.save(model, model_path+'/model.pt')
            input_x = torch.rand(1, 1, input_shape[1], input_shape[0]).to(DEVICE)
            traced_model = traced_func(model, saved_path=model_path+'/model_trace.pt', X=input_x)
        
        if early_stopping.early_stop:
            print("Early Stopping !! ")
            break