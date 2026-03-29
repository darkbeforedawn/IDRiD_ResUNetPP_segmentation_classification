from typing import Literal, Tuple
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights
)

class BackboneFeatureExtractor(nn.Module):
    """ResnetFeatureExtractor base class"""
    def __init__(self, variant: Literal['r18', 'r34', 'r50', 'r101', 'r152'] = 'r50'):
        super().__init__()
        cfg = {
            "r18": (resnet18,  ResNet18_Weights.IMAGENET1K_V1, [64, 64, 128, 256, 512]),
            "r34": (resnet34,  ResNet34_Weights.IMAGENET1K_V1, [64, 64, 128, 256, 512]),
            "r50": (resnet50,  ResNet50_Weights.IMAGENET1K_V2, [64, 256, 512, 1024, 2048]),
            "r101": (resnet101, ResNet101_Weights.IMAGENET1K_V2, [64, 256, 512, 1024, 2048]),
            "r152": (resnet152, ResNet152_Weights.IMAGENET1K_V2, [64, 256, 512, 1024, 2048]),
        }
        ctor, weights, channels = cfg[variant]
        self.model = ctor(weights=weights)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """forward method"""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        c1 = self.model.relu(x)      # stride 2
        x = self.model.maxpool(c1)   # stride 4
        c2 = self.model.layer1(x)    # stride 4
        c3 = self.model.layer2(c2)   # stride 8
        c4 = self.model.layer3(c3)   # stride 16
        c5 = self.model.layer4(c4)   # stride 32
        return c1, c2, c3, c4, c5
