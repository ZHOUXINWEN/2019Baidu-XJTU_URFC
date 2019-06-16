import torch
import torch.nn as nn
from basenet.resnext_features.resnext101_64x4d_features import resnext101_64x4d_features

class ResNeXt101_64x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_64x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_64x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x
