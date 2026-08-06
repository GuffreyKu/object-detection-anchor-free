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


def make_scaler(device, enabled=True):
    """
    Gradient scaler for mixed precision. Measured on an M5 Pro this is ~34% faster
    and halves activation memory. CPU is excluded: fp16 autocast is not a win there.
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
    def __init__(self, optimizer, lr, warmup_len, total_iters):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_len = warmup_len
        self.total_iters = total_iters
        self.current_iter = 0
    
    def get_lr(self):
        if self.current_iter < self.warmup_len:
            lr = self.lr * (self.current_iter + 1) / self.warmup_len
        else:
            cur = self.current_iter - self.warmup_len
            total= self.total_iters - self.warmup_len
            lr = 0.5 * (1 + np.cos(np.pi * cur / total)) * self.lr
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