import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import torch
import torch.nn as nn

from model.backnone import Resnet, EfficientNet, MobileNet
# from backnone import Resnet, EfficientNet, MobileNet


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
        hm = self.cls_head(x)
        wh = self.wh_head(x)
        offset = self.offset_head(x)

        return hm, wh, offset
    
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
    def __init__(self, in_channels, bn_momentum=0.1):
        super(CenterNetDecoder, self).__init__()

        self.in_channels = in_channels

        # h/32, w/32, 2048 -> h/16, w/16, 256 -> h/8, w/8, 128 -> h/4, w/4, 64
        # self.dconv1 = nn.ConvTranspose2d(in_channels=self.in_channels,
        #                                 out_channels=256,
        #                                 kernel_size=4,
        #                                 stride=2,
        #                                 padding=1,
        #                                 output_padding=0,
        #                                 bias=False)
        
        # self.dconv2 = nn.ConvTranspose2d(in_channels=256,
        #                                 out_channels=128,
        #                                 kernel_size=4,
        #                                 stride=2,
        #                                 padding=1,
        #                                 output_padding=0,
        #                                 bias=False)
        
        # self.dconv3 = nn.ConvTranspose2d(in_channels=128,
        #                                 out_channels=64,
        #                                 kernel_size=4,
        #                                 stride=2,
        #                                 padding=1,
        #                                 output_padding=0,
        #                                 bias=False)
        
        # self.dconv4 = nn.ConvTranspose2d(in_channels=64,
        #                                 out_channels=64,
        #                                 kernel_size=4,
        #                                 stride=2,
        #                                 padding=1,
        #                                 output_padding=0,
        #                                 bias=False)
        output_size = 256 
        self.conv4 = nn.Conv2d(self.in_channels, output_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c4 = nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv3 = nn.Conv2d(1024, output_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.c3 = nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv2 = nn.Conv2d(512, output_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv1 = nn.Conv2d(256, output_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
       
        self.output = nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, c1, c2, c3, c4):
        # c4 = [1, 2048, 8, 8], [1, 2048, 16, 16]
        # c3 = [1, 1024, 16, 16], [1, 1024, 32, 32]
        # c2 = [1, 512, 32, 32], [1, 512, 64, 64]
        # c1 = [1, 256, 64, 64], [1, 256, 128, 128]
        x = self.conv4(c4)
        x = self.up4(x)
        x = self.c4(x) + self.conv3(c3)
        x = self.c3(x)
        x = self.up3(x) + self.conv2(c2)
        x = self.up2(x) + self.conv1(c1)
        x = self.up1(x)
        x = self.output(x)


        return x
    
class CenterNet(nn.Module):
    def __init__(self, num_classes=2):
        """
        Args:
            num_classes: int
        """
        super(CenterNet, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, padding=0, bias=False)
        # h, w, 3 -> h/32, w/32, 2048
        # self.backbone = EfficientNet() 
        self.backbone = Resnet()
        # self.backbone = MobileNet()


        # h/32, w/32, 2048 -> h/4, w/4, 64
        self.decoder = CenterNetDecoder(2048)

        # feature height and width: h/4, w/4
        # hm channel: num_classes
        # wh channel: 2
        # offset channel: 2
        self.head = CenterNetHead(in_channel=256, channel=64, num_classes=num_classes)

        # self.centerPool = CenterNetPoolingNMS(kernel=3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head.cls_head[-2].weight.data.fill_(0)
        self.head.cls_head[-2].bias.data.fill_(-2.19)

    def forward(self, x):
        x = self.conv(x)
        c1, c2, c3, c4 = self.backbone(x)
        x = self.decoder(c1, c2, c3, c4)

        hms_pred, whs_pred, offsets_pred = self.head(x)
        # hms_pred = self.centerPool(hms_pred)
        return hms_pred, whs_pred, offsets_pred
        
    
if __name__ == "__main__":
    model = CenterNetPoolingNMS(kernel=3)
    dummy_input = torch.randn(1, 1, 256, 256)
    torch.onnx.export(model, dummy_input, "maxpool_model.onnx")
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, "../savemodel/maxpool.pt")