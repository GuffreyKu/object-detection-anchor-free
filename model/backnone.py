import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
from torchvision import models

# Every backbone here returns four feature maps at strides 4, 8, 16, 32 and declares
# their widths in `out_channels`, because CenterNetDecoder builds its lateral 1x1 convs
# from that tuple. Stride 4 at the finest level is not negotiable: pt_dataset/dataset.py
# hardcodes stride 4 when it encodes targets, and RoIClassifier pools at spatial_scale
# 1/4. A backbone whose finest level is stride 8 does not raise anywhere - it silently
# trains against targets at twice the intended scale.


class Resnet(nn.Module):
    def __init__(self, weights="IMAGENET1K_V1"):
        super().__init__()

        backbone = models.resnet50(weights=weights)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.pool = nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.out_channels = (256, 512, 1024, 2048)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        return l1, l2, l3, l4


class EfficientNet(nn.Module):
    # Last block of each stride. Measured on efficientnet_b3 at 512: blocks 0-1 are
    # stride 2, 2 is stride 4, 3 is stride 8, 4-5 are stride 16, 6-8 are stride 32.
    # These taps used to be {3, 5, 7, 8}, which is strides 8/16/32/32 - no stride-4
    # level at all, and two levels at the same resolution, so the decoder's
    # `up4(x) + conv3(c3)` could not even line up spatially.
    TAPS = (2, 3, 5, 8)

    def __init__(self, weights="IMAGENET1K_V1"):
        super(EfficientNet, self).__init__()
        model = models.efficientnet_b3(weights=weights)
        self.features = model.features
        self.out_channels = (32, 48, 136, 1536)

    def forward(self, x):
        outputs = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self.TAPS:
                outputs.append(x)
        return tuple(outputs)


class MobileNet(nn.Module):
    TAPS = (3, 6, 12, 16)

    def __init__(self, weights="IMAGENET1K_V1"):
        super().__init__()

        model = models.mobilenet_v3_large(weights=weights)
        self.features = model.features
        self.out_channels = (24, 40, 112, 960)

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.TAPS:
                outputs.append(x)
        return tuple(outputs)


class Swin(nn.Module):
    """
    torchvision Swin as a 4-level pyramid at strides 4/8/16/32.

    Swin rather than a plain ViT because its relative position bias is window-relative,
    not absolute: the pretrained 224 weights transfer to 512 with no position-embedding
    surgery, and it emits four resolutions natively instead of one stride-16 token grid.
    A torchvision vit_* would also hard-assert image_size == 224 and fail torch.jit.trace.

    Only `.features` is kept; the classifier tail is dropped so its weights are neither
    carried nor left without gradients under DDP.

    Two things this adds on top of the torchvision module:

    1. NHWC -> NCHW. Every swin stage emits (N, H, W, C) while the decoder, roi_align
       and the heads are all NCHW.

    2. A LayerNorm per output stage. torchvision normalises only after the LAST stage,
       which we discard, so the taps are raw pre-norm residual streams. Measured at 512
       on ImageNet-normalised input, the four taps peak at 10.4 / 10.6 / 304.9 / 151.8
       against resnet50's 3.1 / 3.3 / 1.6 / 8.3 - roughly 37x hotter at c3.
       CenterNetDecoder has no normalisation and no activation anywhere, and its
       four-level sum is the thing that overflowed fp16 at 65504 (see trainer.py's
       use_amp note); handing it 300-magnitude inputs makes that far worse. These four
       norms cost 2880 parameters and put the decoder back in resnet50's range. It is
       also what the reference Swin detection code does - norm0..norm3 feeding the FPN -
       so it is a property of using Swin for dense prediction, not a decoder change.
    """
    # Last block of each stage in `features`; the even indices are the patch merges.
    TAPS = (1, 3, 5, 7)

    def __init__(self, variant="swin_t", weights="IMAGENET1K_V1", norm_out=True,
                 stochastic_depth_prob=None):
        """
        stochastic_depth_prob: torchvision's default is 0.2 for swin_t (12 StochasticDepth
            layers, active in train() only). That is regularisation resnet50 never had,
            and it is the first knob to try if train loss falls while valid mAP lags:
            measured on the 32-image smoke overfit, swin_t reaches reranked mAP 0.1350
            against resnet50's 0.6036 over the same 600 steps. On 12k real images with
            augmentation the tradeoff is likely the other way round, so do not lower it
            on the strength of the smoke number alone. None keeps torchvision's default.
        """
        super().__init__()
        kw = {} if stochastic_depth_prob is None else {"stochastic_depth_prob": stochastic_depth_prob}
        net = getattr(models, variant)(weights=weights, **kw)
        self.features = net.features
        # 96 for swin_t/s, 128 for swin_b, and each stage doubles it. Derived rather
        # than tabulated so every swin variant works without another lookup table.
        embed = net.features[0][0].out_channels
        self.out_channels = tuple(embed * 2 ** i for i in range(4))
        self.norms = (nn.ModuleList([nn.LayerNorm(c) for c in self.out_channels])
                      if norm_out else None)

    def forward(self, x):
        outputs = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self.TAPS:
                y = x if self.norms is None else self.norms[len(outputs)](x)
                # contiguous(): everything downstream indexes this as NCHW, and a bare
                # permute leaves a view whose stride order confuses roi_align.
                outputs.append(y.permute(0, 3, 1, 2).contiguous())
        return tuple(outputs)


BACKBONES = {
    "resnet50":           lambda **kw: Resnet(**kw),
    "efficientnet_b3":    lambda **kw: EfficientNet(**kw),
    "mobilenet_v3_large": lambda **kw: MobileNet(**kw),
    "swin_t":             lambda **kw: Swin("swin_t", **kw),
    "swin_s":             lambda **kw: Swin("swin_s", **kw),
    "swin_b":             lambda **kw: Swin("swin_b", **kw),
    "swin_v2_t":          lambda **kw: Swin("swin_v2_t", **kw),
    "swin_v2_s":          lambda **kw: Swin("swin_v2_s", **kw),
    "swin_v2_b":          lambda **kw: Swin("swin_v2_b", **kw),
}


def build_backbone(name, **kwargs):
    """
    Resolve a backbone name from BACKBONES, or pass through a ready-made nn.Module.

    The module escape hatch is what lets a test inject a tiny stub with
    out_channels=(8,16,32,64) and exercise the decoder without downloading 100 MB of
    ImageNet weights.
    """
    if isinstance(name, nn.Module):
        if not hasattr(name, "out_channels"):
            raise TypeError(
                "a custom backbone must expose out_channels=(c1, c2, c3, c4) for its "
                "four feature maps at strides 4/8/16/32")
        return name
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; choose one of {sorted(BACKBONES)} "
                       f"or pass an nn.Module exposing out_channels")
    return BACKBONES[name](**kwargs)


if __name__ == "__main__":
    # Doubles as the topology test: every backbone must report four levels at strides
    # 4/8/16/32 whose widths match its declared out_channels.
    size = 512
    x = torch.randn(1, 3, size, size)

    for name in BACKBONES:
        model = BACKBONES[name](weights=None).eval()
        with torch.no_grad():
            outs = model(x)
        strides = [size // t.shape[-1] for t in outs]
        widths = tuple(t.shape[1] for t in outs)
        ok = strides == [4, 8, 16, 32] and widths == tuple(model.out_channels)
        print(f"{'ok ' if ok else 'BAD'} {name:20} strides {strides} "
              f"channels {widths} declared {tuple(model.out_channels)}")
