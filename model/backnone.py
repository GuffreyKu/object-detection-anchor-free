import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
from torchvision import models

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