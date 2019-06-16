from torch import nn
import torch
from torchvision import models,transforms,datasets
import torch.nn.functional as F
from senet import se_resnet50,se_resnext101_32x4d,se_resnext50_32x4d, se_resnext26_32x4d

class multiscale_se_resnext_cat(nn.Module):
    def __init__(self,num_class):
        super(multiscale_se_resnext_cat,self).__init__()

        self.base_model1 = se_resnext50_32x4d(9, None)
        self.base_model2 = se_resnext50_32x4d(9, None)

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(4096, 1024)

    def forward(self, x):
        input_size = x.size()[2]
        self.interp = nn.UpsamplingBilinear2d(size = (int(input_size*0.75)+1,  int(input_size*0.75)+1))

        x2 = self.interp(x)

        x = self.base_model1(x)

        x2 = self.base_model2(x2)

        out = torch.cat((x, x2),1)   #x + x2

        out = self.dropout(out)
        out = self.classifier(out)

        return out
