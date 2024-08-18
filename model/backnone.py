import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
from torchvision import models

def activation_func(activation):
    return nn.ModuleDict({
        'selu': nn.SELU(inplace=True),
        'relu': nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
        'sigmoid': nn.Sigmoid(),
        'prelu': nn.PReLU(),
        'softmax': nn.Softmax(dim=1),
        'gelu': nn.GELU()})[activation]

class DWCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, stride=1):
        super(DWCNNBlock, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=in_ch,
                                    bias=False)
        
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=False)
        
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self,input):
        out = self.relu6(self.depth_conv(input))
        out = self.point_conv(out)
        return out

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

        # self.features = nn.Sequential(*list(backbone.children())[:-2])
        
    def forward(self, x):
        # x = self.features(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        return l1, l2, l3, l4

class MobileNet(nn.Module):
    def __init__(self, weights="IMAGENET1K_V1"):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights=weights)
        print(backbone)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
    def forward(self, x):
        x = self.features(x)
        return x

class EfficientNet(nn.Module):
    def __init__(self, weights="IMAGENET1K_V1"):
        super(EfficientNet, self).__init__()
        model = models.efficientnet_b3(weights=weights)
        self.features = model.features
        # print(len(self.features))
    def forward(self, x):
        outputs = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in {3, 5, 7, 8}:  # Adjust these indices to capture desired intermediate outputs
                outputs.append(x)
        return outputs

class MobileNet(nn.Module):
    def __init__(self, weights="IMAGENET1K_V1"):
        super().__init__()

        model = models.mobilenet_v3_large(weights=weights)
        self.features = model.features
        
    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in {3, 6, 12, 16}:  # Adjust these indices to capture desired intermediate outputs
                outputs.append(x)
        return outputs
     
if __name__ == "__main__":

    from torchinfo import summary
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input = torch.randn(1, 3, 512, 512).to("cpu")

    # print(model)
    model = Resnet().to("cpu")
    output1, output2, output3, output4 = model(input)
    print(output1.size())# 1, 256, 128, 128
    print(output2.size())# 1, 512, 64, 64
    print(output3.size())# 1, 1024, 32, 32
    print(output4.size())# 1, 2048, 16, 16

    # print(model)
    # summary(model, input_data=input)