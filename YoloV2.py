import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn
from torchinfo import summary

def conv_unit(num_filters, kernel):
    layers = []
    layers.append(nn.Conv2d(num_filters, num_filters, kernel_size = kernel, padding = 'same'))
    layers.append(nn.BatchNorm2d(num_filters))
    layers.append(nn.ReLU())
    return layers

class YoloV2(nn.Module):
    def __init__(self) -> None:
        super(YoloV2, self).__init__()
        model = vgg16_bn(weights = 'IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(model.children()))[:-2]
        self.layer1 = nn.Sequential(*conv_unit(512, 3))
        self.layer2 = nn.Sequential(*conv_unit(512, 3))
        self.layer3 = nn.Sequential(*conv_unit(512, 3))
        self.output = nn.Conv2d(512, 50, kernel_size = 1)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)

        return x

model = YoloV2()
summary(model, input_size = (1, 3, 224, 224))