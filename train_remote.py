"""Train on a remote GPU box while the dataset stays on this machine.

torch.distributed.rpc splits it in two: the GPU box runs the whole training loop
(model, optimiser, loss) and calls back here for every batch. Nothing but tensors
crosses the wire, and checkpoints are shipped back so they land next to the data.

    # on the GPU box (rank 0, also the rendezvous master)
    python train_remote.py --role gpu --addr <gpu-ip>

    # here
    python train_remote.py --role data --addr <gpu-ip>

Both machines need this repo checked out and matching torch versions. They also
need to reach each other directly - TensorPipe opens its own sockets both ways,
so a plain `ssh -L` on the master port is not enough. Same LAN, VPN or Tailscale.
"""
import argparse
import io
import os

import torch
import torch.distributed.rpc as rpc
import torch_optimizer as optim_alg

# Single source of truth for the hyperparameters: trainer.py guards its loop with
# __main__, so importing it just gets the constants.
from trainer import (annotation_path, batch_size, epochs, input_shape, model_path,
                     use_amp, valid_ratio)
from check_remote import net_setup
from flow.flow import evaluate, train
from model.centerNet import CenterNet
from model.loss import TotalLoss
from pt_dataset.dataloader import dataloader
from utils.pytorchtools import (CosineDecayWarmup, EarlyStopping, assert_finite,
                                get_device, make_scaler)
from utils.tool import folderCheck, load_annotation, stratified_split

GPU, DATA = "gpu", "data"  # rank 0, rank 1


# --------------------------------------------------------------------------
# runs on the data machine, called by the GPU box
# --------------------------------------------------------------------------
_loaders, _iters, _meta = {}, {}, None


def meta():
    return _meta


def loader_len(split):
    return len(_loaders[split])


def reset(split):
    _iters[split] = iter(_loaders[split])


def next_batch(split, fp16):
    try:
        b = next(_iters[split])
    except StopIteration:
        return None
    # ponytail: fp16 on the wire halves the ~100MB/batch. Images and targets are
    # normalised or small integers so the range is safe; boxes go as-is because
    # fp16 only resolves to ~0.5px at 512. Pass --fp32-wire if you want it off.
    stacked = [t.half() if fp16 else t for t in b[:5]]
    return (*stacked, b[5], b[6])


def save_bytes(name, blob):
    path = os.path.join(model_path, name)
    with open(path, "wb") as f:
        f.write(blob)
    print(f"saved {path} ({len(blob) / 1e6:.1f} MB)")


# --------------------------------------------------------------------------
# runs on the GPU box
# --------------------------------------------------------------------------
class RemoteLoader:
    """Drop-in for a DataLoader whose batches are built on the data machine."""

    def __init__(self, split, fp16=True):
        self.split, self.fp16 = split, fp16
        self.n = rpc.rpc_sync(DATA, loader_len, (split,))

    def __len__(self):
        return self.n

    def __iter__(self):
        rpc.rpc_sync(DATA, reset, (self.split,))
        fut = rpc.rpc_async(DATA, next_batch, (self.split, self.fp16))
        while True:
            b = fut.wait()
            if b is None:
                return
            # Ask for the next batch before handing this one over, so the transfer
            # overlaps with the GPU step instead of stalling it.
            fut = rpc.rpc_async(DATA, next_batch, (self.split, self.fp16))
            yield (*[t.float() for t in b[:5]], b[5], b[6])


def ship(model, name):
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, buf)
    rpc.rpc_sync(DATA, save_bytes, (name, buf.getvalue()))


def run_training(fp16_wire):
    DEVICE = get_device()
    num_classes, class_names = rpc.rpc_sync(DATA, meta, ())
    print(f"training on {DEVICE}, {num_classes} classes")

    train_loader = RemoteLoader("train", fp16_wire)
    valid_loader = RemoteLoader("valid", fp16_wire)

    criterion = TotalLoss().to(DEVICE)
    model = CenterNet(num_classes=num_classes).to(DEVICE)
    optimizer = optim_alg.Ranger(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scaler = make_scaler(DEVICE, enabled=use_amp)
    early_stopping = EarlyStopping(patience=30, verbose=False)
    scheduler = CosineDecayWarmup(optimizer=optimizer,
                                  lr=1e-3,
                                  warmup_len=int(epochs * 0.1) * len(train_loader),
                                  total_iters=epochs * len(train_loader))

    best = 0
    for e in range(epochs):
        train(now_ep=e, model=model, optimizer=optimizer, scheduler=scheduler,
              dataloader=train_loader, criterion=criterion, DEVICE=DEVICE, scaler=scaler)

        assert_finite(model)
        b_valid_loss, map1, map2, ev = evaluate(mode="valid",
                                                model=model,
                                                dataloader=valid_loader,
                                                criterion=criterion,
                                                DEVICE=DEVICE,
                                                image_size=input_shape,
                                                amp=scaler.is_enabled(),
                                                num_classes=num_classes)

        print(f"epoch {e}: loss {b_valid_loss:.3f}  mAP@0.5 {map1:.4f}" + (f"  reranked {map2:.4f}" if map2 is not None else ""))
        early_stopping(b_valid_loss)

        # Stage-1, matching trainer.py: the rerank measured 4.5 points worse.
        if map1 >= best:
            best = map1
            print(ev.report(class_names))
            # ponytail: state_dict only. The torchscript trace trainer.py also writes
            # would be pinned to the remote's CUDA device; run torch2onnx.py locally
            # off this checkpoint instead.
            ship(model, "model.pth")

        if early_stopping.early_stop:
            print("Early Stopping !! ")
            break


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=[GPU, DATA], required=True)
    p.add_argument("--addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
                   help="address of the GPU box (rank 0), same value on both sides")
    p.add_argument("--port", type=int, default=int(os.environ.get("MASTER_PORT", 29500)))
    p.add_argument("--fp32-wire", action="store_true", help="do not compress batches to fp16")
    p.add_argument("--timeout", type=int, default=300,
                   help="seconds before an RPC gives up. A ~50MB fp16 batch needs this "
                        "well above the 60s default, but not so high that a dead link "
                        "looks like a slow one - run check_remote.py if it trips.")
    p.add_argument("--iface", default=None,
                   help="network interface facing the peer, e.g. ppp0 / utun3 / eth0. "
                        "Needed on a multi-homed host - see check_remote.net_setup().")
    args = p.parse_args()
    net_setup(args.addr, args.iface)

    opts = rpc.TensorPipeRpcBackendOptions(
        init_method=f"tcp://{args.addr}:{args.port}",
        rpc_timeout=args.timeout,
        num_worker_threads=16,
    )

    if args.role == DATA:
        global _meta
        folderCheck([model_path])
        annotations, class_names = load_annotation(annotation_path)
        _meta = (len(class_names), class_names)
        tr, va = stratified_split(annotations, valid_ratio=valid_ratio)
        print(f"{len(class_names)} classes, {len(tr)} train / {len(va)} valid images")
        _loaders["train"], _loaders["valid"] = dataloader(
            tr, va, len(class_names), batch_size, input_shape)

        rpc.init_rpc(DATA, rank=1, world_size=2, rpc_backend_options=opts)
        print("serving batches, waiting for the GPU box ...")
        rpc.shutdown()  # keeps serving until the GPU box finishes and shuts down too
    else:
        rpc.init_rpc(GPU, rank=0, world_size=2, rpc_backend_options=opts)
        try:
            run_training(fp16_wire=not args.fp32_wire)
        finally:
            rpc.shutdown()


if __name__ == "__main__":
    main()
