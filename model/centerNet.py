import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import torch
import torch.nn as nn
from torchvision.ops import roi_align

from model.backnone import build_backbone


class CenterNetHead(nn.Module):
    def __init__(self, num_classes=80, in_channel=64, channel=64, bn_momentum=0.1):
        super(CenterNetHead, self).__init__()

        # heatmap
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channel, channel, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.Mish(inplace=True),
            nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        # bounding boxes height and width
        self.wh_head = nn.Sequential(
            nn.Conv2d(in_channel, channel, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.Mish(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0)
            )
        # center point offset
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channel, channel, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.Mish(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        # fp32 regardless of autocast. These heads regress raw box sizes out of an
        # unnormalised decoder sum, and that feeds BatchNorms whose running stats are
        # updated in the forward pass, where GradScaler cannot protect them (see
        # trainer.py's use_amp note). Under fp16 this was only a partial mitigation - it
        # moved the first failure from step 29 to step 225 but could not prevent it,
        # because the inf arrived already formed; bf16 is what actually prevents it.
        # Kept because it costs nothing - three convs at stride 4 next to a ResNet50 -
        # and running_var is a square, so it is the one place worth keeping full range.
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            hm = self.cls_head(x)
            wh = self.wh_head(x)
            offset = self.offset_head(x)

        return hm, wh, offset
    
class RoIClassifier(nn.Module):
    def __init__(self, num_classes, in_channel=256, channel=256, out_size=7, stride=4):
        """
        Second stage: pool the shared feature map over each box and classify it.

        The detector reads its class from the single feature-map cell at the box
        center, while a median box spans ~29 cells. This pools the whole extent
        instead. Measured on ground-truth crops (DATASET.md 14.2), a dedicated
        classifier reaches 73.6% on the 9 confusable container classes at the detail
        the stride-4 map already carries, and re-cropping from the original image at
        full resolution only adds ~3 points - which is why this reuses the feature
        map rather than running a second backbone over N crops.
        """
        super().__init__()
        self.out_size = out_size
        self.spatial_scale = 1.0 / stride
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Mish(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, num_classes),
        )

    def forward(self, feat, rois):
        '''feat: (n, c, h, w) decoder output. rois: list of (Ni, 4) xyxy in input pixels.'''
        # fp32 regardless of autocast: MPS has no fp16 roi_align backward, and boxes
        # must match the feature dtype anyway. This head is a few hundred boxes wide,
        # so keeping it out of autocast costs nothing measurable.
        with torch.autocast(feat.device.type, enabled=False):
            feat = feat.float()
            rois = [r.float() for r in rois]
            pooled = roi_align(feat, rois, output_size=self.out_size,
                               spatial_scale=self.spatial_scale, aligned=True)
            if pooled.shape[0] == 0:
                return pooled.new_zeros((0, self.net[-1].out_features))
            return self.net(pooled)


class CenterNetPoolingNMS(nn.Module):
    def __init__(self, kernel=3):
        """
        To replace traditional nms method. Input is heatmap, the num of channel is num_classes,
        So one object center has strongest response, where use torch.max(heatmap, dim=-1), it only
        filter single pixel max value, the neighbour pixel still have strong response, so we should
        use max pooling stride=1 to filter this fake center point.
        Args:
            kernel: max pooling kernel size
        """
        super(CenterNetPoolingNMS, self).__init__()
        self.pad = (kernel - 1) // 2
        self.max_pool = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=(kernel - 1) // 2)

    def forward(self, x):
        xmax = self.max_pool(x)
        keep = (xmax == x).float()

        return x * keep
    
class CenterNetDecoder(nn.Module):
    def __init__(self, in_channels, output_size=256, bn_momentum=0.1):
        """
        in_channels: (c1, c2, c3, c4), the backbone's four widths at strides 4/8/16/32.
                     Pass backbone.out_channels - (256, 512, 1024, 2048) for resnet50,
                     (96, 192, 384, 768) for swin_t.

        bn_momentum is unused: this decoder deliberately has no normalisation at all.
        Kept so the signature does not change under callers that still pass it.
        """
        super(CenterNetDecoder, self).__init__()

        if isinstance(in_channels, int):
            raise TypeError(
                "CenterNetDecoder takes the backbone's four lateral widths, not just "
                "c4. Pass backbone.out_channels, e.g. (256, 512, 1024, 2048) for "
                "resnet50 or (96, 192, 384, 768) for swin_t.")

        c1, c2, c3, c4 = self.in_channels = tuple(in_channels)

        # stride 32 -> 16 -> 8 -> 4. The lateral widths come from the backbone; the rest
        # of this, including the absence of any norm or activation, is unchanged.
        self.conv4 = nn.Conv2d(c4, output_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c4 = nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv3 = nn.Conv2d(c3, output_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.c3 = nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv2 = nn.Conv2d(c2, output_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv2d(c1, output_size, kernel_size=1, stride=1, padding=0, bias=False)

        self.output = nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, c1, c2, c3, c4):
        # Shapes for a 512x512 input, resnet50 / swin_t:
        # c4 = [1, 2048, 16, 16] / [1,  768, 16, 16]
        # c3 = [1, 1024, 32, 32] / [1,  384, 32, 32]
        # c2 = [1,  512, 64, 64] / [1,  192, 64, 64]
        # c1 = [1, 256,128,128]  / [1,   96,128,128]
        x = self.conv4(c4)
        x = self.up4(x)
        x = self.c4(x) + self.conv3(c3)
        x = self.c3(x)
        x = self.up3(x) + self.conv2(c2)
        x = self.up2(x) + self.conv1(c1)
        # Stops at c1's resolution: output stride 4. Measured on this dataset, going
        # finer than /4 buys nothing (2 boxes out of 32189 collide at /4, 0 at /2)
        # and quadruples the heatmap memory.
        x = self.output(x)

        return x
    
class CenterNet(nn.Module):
    def __init__(self, num_classes=2, backbone="swin_t", roi_head=True):
        """
        Args:
            num_classes: int
            backbone: a name from model.backnone.BACKBONES, or a ready nn.Module
                      exposing out_channels=(c1, c2, c3, c4) at strides 4/8/16/32.
                      Default swin_t: +5.3 ImageNet points over resnet50, for a longer
                      step and more activation memory (see README).
                      Backbone weights are NOT transferable between choices - the
                      checkpoints under savemodel/ belong to whichever one wrote them,
                      and utils.pytorchtools says so by name when they do not match.
            roi_head: add the second-stage RoI classifier
        """
        super(CenterNet, self).__init__()

        # RGB in. The old 1->3 conv adapter existed only to feed grayscale to an
        # ImageNet backbone; with real colour it just discards information.
        self.backbone_name = backbone if isinstance(backbone, str) else type(backbone).__name__
        self.backbone = build_backbone(backbone)

        if len(self.backbone.out_channels) != 4:
            raise ValueError(
                f"{self.backbone_name} declares {len(self.backbone.out_channels)} "
                f"feature levels; the decoder needs exactly 4, at strides 4/8/16/32")

        # four levels -> one map at stride 4, 256 channels. pt_dataset/dataset.py and
        # RoIClassifier below both hardcode that 4.
        self.decoder = CenterNetDecoder(self.backbone.out_channels)

        # feature height and width: h/4, w/4
        # hm channel: num_classes
        # wh channel: 2
        # offset channel: 2
        self.head = CenterNetHead(in_channel=256, channel=64, num_classes=num_classes)

        self.roi_head = RoIClassifier(num_classes, in_channel=256) if roi_head else None

        # self.centerPool = CenterNetPoolingNMS(kernel=3)
        # Only the from-scratch parts. Running this over self.modules() would also hit the
        # backbone and throw away the ImageNet weights we just loaded.
        for m in [*self.decoder.modules(), *self.head.modules(),
                  *(self.roi_head.modules() if roi_head else [])]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head.cls_head[-2].weight.data.fill_(0)
        self.head.cls_head[-2].bias.data.fill_(-2.19)

    def forward(self, x, rois=None):
        '''rois: list of (Ni, 4) xyxy boxes in input pixels, one entry per image.

        Without rois this returns the three detection heads, which keeps torch.jit.trace
        and the onnx export unchanged. With rois it also returns the RoI logits.
        '''
        c1, c2, c3, c4 = self.backbone(x)
        feat = self.decoder(c1, c2, c3, c4)

        hms_pred, whs_pred, offsets_pred = self.head(feat)
        # hms_pred = self.centerPool(hms_pred)
        if rois is None or self.roi_head is None:
            return hms_pred, whs_pred, offsets_pred
        return hms_pred, whs_pred, offsets_pred, self.roi_head(feat, rois)
        
    
if __name__ == "__main__":
    model = CenterNetPoolingNMS(kernel=3)
    dummy_input = torch.randn(1, 1, 256, 256)
    torch.onnx.export(model, dummy_input, "maxpool_model.onnx")
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, "../savemodel/maxpool.pt")