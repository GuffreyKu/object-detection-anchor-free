import os

import numpy as np
import torch


def get_device(verbose=True):
    """
    Pick the best available accelerator: CUDA, then Apple Metal (MPS), then CPU.

    Returns a torch.device. cudnn.benchmark is enabled only on CUDA, where it means
    something.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        name = "Apple Metal (MPS)"
    else:
        device = torch.device("cpu")
        name = "CPU"

    if verbose:
        print(f"device: {device} ({name})")
    return device


def amp_dtype(device):
    """
    Autocast dtype for this device: bfloat16 where the hardware has it, else float16.

    This is the fix for the BatchNorm poisoning described in trainer.py. That failure
    was a *range* problem, not a precision one: CenterNetDecoder sums four FPN levels
    with no normalisation, the sum eventually passes fp16's 65504 ceiling, and the
    resulting inf lands in the head BatchNorms' running stats, which are updated in the
    forward pass and so are not covered by GradScaler. bf16 keeps fp32's 8-bit exponent
    (max ~3.4e38), so the same activations stay finite; it pays for that with a 8-bit
    mantissa, which the fp32 loss and fp32 heads already absorb.

    bf16 needs Ampere or newer. is_bf16_supported() is the check rather than a compute
    capability comparison because it also covers the ROCm and emulation cases.
    """
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_scaler(device, enabled=True):
    """
    Gradient scaler for mixed precision. Measured on an M5 Pro this is ~34% faster
    and halves activation memory. CPU is excluded: fp16 autocast is not a win there.

    Left enabled under bf16 too, where it is not strictly required (bf16 gradients have
    fp32's range, so they do not underflow the way fp16's do). It is kept because
    scaler.step() skips any step whose gradients are inf or nan, which is a free guard
    on a model whose wh head regresses unbounded box sizes, and because it keeps a
    single flag - scaler.is_enabled() - meaning "mixed precision is on" for both dtypes.
    """
    use = enabled and device.type in ("cuda", "mps")
    return torch.amp.GradScaler(device.type, enabled=use)


def assert_finite(model):
    """
    Fail fast if any buffer has gone non-finite.

    BatchNorm running_mean / running_var are updated in the forward pass, so
    GradScaler cannot protect them from an fp16 overflow the way it protects
    parameters. Once poisoned they never recover, and because train() uses batch
    statistics the damage is invisible until eval(). Checking costs nothing;
    finding out three hours into a run costs the run.
    """
    bad = [n for n, b in model.named_buffers() if not torch.isfinite(b).all()]
    assert not bad, f"non-finite buffers, training is corrupted: {bad[:5]}"


def describe_backbone(state_dict):
    """
    Name the backbone a bare state_dict was trained with, by fingerprinting its keys.

    Bare state_dicts carry no metadata, and a backbone mismatch is otherwise reported as
    several hundred missing/unexpected key names with the diagnosis nowhere in sight.
    """
    if "backbone.layer4.0.conv1.weight" in state_dict:
        return "resnet50"
    if any(k.endswith("attn.relative_position_bias_table") for k in state_dict):
        embed = state_dict["backbone.features.0.0.weight"].shape[0]
        v2 = any("cpb_mlp" in k for k in state_dict)
        return ("swin_v2_" if v2 else "swin_") + ("b" if embed == 128 else "t or swin_s")
    c4 = state_dict.get("decoder.conv4.weight")
    if c4 is not None:
        return f"unknown (its decoder expects a {c4.shape[1]}-channel c4)"
    return "unknown"


def load_weights(net, state_dict, path):
    """
    load_state_dict, but turn a backbone mismatch into one readable paragraph.

    Backbone weights do not transfer between architectures, so this is a dead end rather
    than something to load partially - the only useful thing to do is say which two
    things disagree and what the two ways out are.
    """
    try:
        net.load_state_dict(state_dict)
    except RuntimeError as err:
        got = describe_backbone(state_dict)
        want = getattr(net, "backbone_name", type(net.backbone).__name__)
        if got == want:
            raise
        first = next((ln.strip() for ln in str(err).splitlines()[1:] if ln.strip()), "")
        raise RuntimeError(
            f"{path} does not fit this model.\n"
            f"    checkpoint backbone : {got}\n"
            f"    this run's backbone : {want}\n"
            f"  Backbone weights are not transferable, so either\n"
            f"    - continue the old run:  CenterNet(num_classes=..., backbone='{got}')\n"
            f"    - or start a {want} run: set `resume = None` in trainer.py, having\n"
            f"      moved savemodel/model.pth and savemodel/last.pth aside first - the\n"
            f"      first improvement overwrites them.\n"
            f"  torch said: {first[:110]}"
        ) from None


def checkpoint_guard(paths, backbone_name):
    """
    Refuse to start a run that would overwrite another backbone's checkpoints.

    trainer.py rewrites last.pth every epoch and model.pth on every improvement, and the
    first improvement over best=0 arrives in epoch 0. So switching backbone and pressing
    go destroys the previous run before it produces anything comparable to replace it.
    Reading two headers costs milliseconds; the files it protects are hours each.
    """
    for path in paths:
        if not os.path.exists(path):
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        found = (blob.get("backbone") if isinstance(blob, dict) and "model" in blob
                 else None) or describe_backbone(
                     blob["model"] if isinstance(blob, dict) and "model" in blob else blob)
        if found == backbone_name or found.startswith("unknown"):
            continue
        # swin_t and swin_s are indistinguishable by key shape alone.
        if backbone_name in found.split(" or "):
            continue
        raise SystemExit(
            f"\n{path} belongs to a {found} run, but this run is configured for "
            f"{backbone_name}.\nStarting would overwrite it: last.pth is rewritten every "
            f"epoch and model.pth on the\nfirst improvement, which happens in epoch 0.\n\n"
            f"  keep the old run    : mv savemodel savemodel_{found}\n"
            f"  or continue it      : set backbone = '{found}' in trainer.py\n"
            f"  or discard it       : rm {' '.join(paths)}\n")


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, early_stopping,
                    best, train_losses, valid_losses, map_history=None):
    """
    Everything needed to resume, overwritten every epoch.

    Deliberately not model.pth. That file is the best-mAP snapshot and is a bare
    state_dict because inference.py and torch2onnx.py load it that way; this one is the
    LAST epoch. Resuming from the best-mAP file instead would pair those weights with an
    optimizer state and a point on the LR curve that never belonged to them.

    Written to a temp file and renamed, because the write takes a second or two and a
    crash partway through would otherwise leave no usable checkpoint at all - rename is
    atomic within a filesystem, so the old one survives until the new one is complete.
    """
    net = model.module if hasattr(model, "module") else model
    tmp = path + ".tmp"
    torch.save({
        "epoch": epoch,
        "backbone": getattr(net, "backbone_name", None),
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "early_stopping": early_stopping.state_dict(),
        "best": best,
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        # The mAP curve. Run 1's was unrecoverable afterwards because only the losses
        # were stored, which made "was it still improving when it stopped?" unanswerable.
        "map_history": map_history if map_history is not None else [],
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    early_stopping=None, map_location=None):
    """
    Restore a run. Returns (start_epoch, best, train_losses, valid_losses).

    Takes either format. A full checkpoint from save_checkpoint() restores the optimizer
    momentum, the LR position and the early-stopping counter as well as the weights. A
    bare state_dict - savemodel/model.pth, or anything train_remote.py shipped - has only
    weights in it, so it can warm-start but not resume: it returns epoch 0 and leaves the
    optimizer and the LR schedule at their initial state. That distinction matters,
    because continuing a cosine run with a freshly reset LR quietly undoes progress.
    """
    # Our own file, and it holds python lists and floats as well as tensors.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    net = model.module if hasattr(model, "module") else model

    if not (isinstance(ckpt, dict) and "model" in ckpt):
        load_weights(net, ckpt, path)
        print(f"resume: {path} holds weights only -> warm start. Optimizer state and LR "
              f"schedule begin from scratch; epoch counter restarts at 0.")
        return 0, 0.0, [], [], []

    # Named up front when the checkpoint recorded it, so the mismatch is reported before
    # several hundred key names are. Files written before this field existed fall through
    # to the fingerprint in load_weights.
    want = getattr(net, "backbone_name", None)
    saved = ckpt.get("backbone")
    if saved is not None and want is not None and saved != want:
        raise RuntimeError(
            f"{path} was trained with backbone {saved!r}, this run uses {want!r}. "
            f"Backbone weights are not transferable: either build the model with "
            f"backbone={saved!r}, or set `resume = None` after moving that file aside.")

    load_weights(net, ckpt["model"], path)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if early_stopping is not None and ckpt.get("early_stopping") is not None:
        early_stopping.load_state_dict(ckpt["early_stopping"])

    start = ckpt["epoch"] + 1
    print(f"resume: {path} finished epoch {ckpt['epoch']} -> continuing at {start}, "
          f"best mAP so far {ckpt.get('best', 0.0):.4f}")
    return (start, ckpt.get("best", 0.0),
            ckpt.get("train_losses", []), ckpt.get("valid_losses", []),
            ckpt.get("map_history", []))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func

    def state_dict(self):
        return {"counter": self.counter, "best_score": self.best_score,
                "early_stop": self.early_stop}

    def load_state_dict(self, state):
        # The counter has to survive a resume, otherwise every restart hands the run
        # another `patience` epochs of not improving and the stop never fires.
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.early_stop = state["early_stop"]

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class CosineDecayWarmup:
    def __init__(self, optimizer, lr, warmup_len, total_iters, min_lr=0.0):
        """
        min_lr: floor the cosine lands on at total_iters. 0.0 reproduces the old curve.
                A small non-zero floor (1e-5 against a 1e-3 peak) keeps the last epochs
                doing something instead of freezing.
        """
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_len = warmup_len
        self.total_iters = total_iters
        self.min_lr = min_lr
        self.current_iter = 0

    def state_dict(self):
        return {"current_iter": self.current_iter, "warmup_len": self.warmup_len,
                "total_iters": self.total_iters, "lr": self.lr, "min_lr": self.min_lr}

    def load_state_dict(self, state):
        """
        Restore the position in the curve, but keep the shape from the current config.

        Only current_iter is taken back. warmup_len / total_iters / lr are left as this
        run configured them, so editing `epochs` between runs reshapes what is left of
        the cosine instead of being silently overridden by the old run's numbers - it
        just says so, because the resulting LR will not match the original curve.
        """
        old = (state.get("warmup_len"), state.get("total_iters"), state.get("lr"),
               state.get("min_lr", 0.0))
        new = (self.warmup_len, self.total_iters, self.lr, self.min_lr)
        if old != new:
            print(f"  LR schedule differs from the checkpoint "
                  f"(warmup/total/peak {old} -> {new}); keeping this run's shape")
        self.current_iter = state["current_iter"]

    def get_lr(self):
        if self.current_iter < self.warmup_len:
            lr = self.lr * (self.current_iter + 1) / self.warmup_len
        else:
            cur = self.current_iter - self.warmup_len
            total = self.total_iters - self.warmup_len
            # Clamp at 1.0. Past total_iters the raw cosine turns around and climbs back
            # toward the peak, so an overrun - or a deadline reshape that lands short -
            # would silently undo the anneal instead of holding at the floor.
            frac = min(1.0, cur / total) if total > 0 else 1.0
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * frac))
        return lr
    
    def step(self):
        lr = self.get_lr()
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        self.current_iter += 1

def traced_func(model, saved_path, X):
    traced_model = torch.jit.trace(model, X)
    torch.jit.save(traced_model, saved_path)
    return traced_model


def save_crop_classifier_checkpoint(path, model, optimizer, scheduler, epoch, best_map,
                                    class_names, input_size=224, crop_expand=0.1,
                                    proposal_topk=100, arc_margin=False,
                                    early_stopping=None):
    """Atomically save the stage-2 classifier and its preprocessing contract."""
    tmp = str(path) + ".tmp"
    torch.save({
        "epoch": epoch,
        "best_map": best_map,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "class_names": list(class_names),
        "background_index": len(class_names),
        "input_size": input_size,
        "crop_expand": crop_expand,
        "proposal_topk": proposal_topk,
        # Which head shape model.state_dict() needs on load - see model/centerNet.py:
        # CropClassifier(arc_margin=...). Missing on checkpoints saved before this existed
        # -> .get(..., False) below reconstructs the plain nn.Linear head they actually have.
        "arc_margin": arc_margin,
        "early_stopping": early_stopping.state_dict() if early_stopping is not None else None,
    }, tmp)
    os.replace(tmp, path)


def load_crop_classifier_checkpoint(path, device, expected_class_names=None):
    """Load and validate the classifier label/preprocessing contract."""
    from model.centerNet import CropClassifier

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    if expected_class_names is not None and list(expected_class_names) != class_names:
        raise RuntimeError("classifier class order does not match the annotation file")
    if checkpoint.get("background_index") != len(class_names):
        raise RuntimeError("classifier checkpoint has an invalid background index")
    model = CropClassifier(len(class_names), pretrained=False,
                           arc_margin=checkpoint.get("arc_margin", False)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
