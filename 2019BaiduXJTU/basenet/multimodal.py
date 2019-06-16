from collections import OrderedDict
import math
import torch
import torch.nn as nn
from torch.utils import model_zoo

from senet import se_resnet50,se_resnext101_32x4d,se_resnext50_32x4d, se_resnext26_32x4d, se_resnet50
#from oct_resnet import oct_resnet26,oct_resnet101
from nasnet import nasnetalarge
from multiscale_resnet import multiscale_resnet
from basenet.multiscale_se_resnext import multiscale_se_resnext
from basenet.multiscale_se_resnext_cat import multiscale_se_resnext_cat
from DPN import DPN92, DPN26
from SKNet import SKNet101
from basenet.multiscale_se_resnext_HR import multiscale_se_resnext_HR

class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MultiModalNet(nn.Module):
    def __init__(self, backbone1, backbone2, drop, pretrained=True):
        super(MultiModalNet, self).__init__()

       
        self.visit_model = DPN26()
        if backbone1 == 'se_resnext101_32x4d' :
            self.img_encoder = se_resnext101_32x4d(9, None)
            self.img_fc = nn.Linear(2048, 256)

        elif backbone1 == 'se_resnext50_32x4d' :
            self.img_encoder = se_resnext50_32x4d(9, None)

            print("load pretrained model from /home/zxw/2019BaiduXJTU/se_resnext50_32x4d-a260b3a4.pth")
            state_dict = torch.load('/home/zxw/2019BaiduXJTU/se_resnext50_32x4d-a260b3a4.pth')

            state_dict.pop('last_linear.bias')
            state_dict.pop('last_linear.weight')
            self.img_encoder.load_state_dict(state_dict, strict = False)

            self.img_fc = nn.Linear(2048, 256)

        elif backbone1 == 'se_resnext26_32x4d' :
            self.img_encoder = se_resnext26_32x4d(9, None)
            self.img_fc = nn.Linear(2048, 256)

        elif backbone1 == 'multiscale_se_resnext' :
            self.img_encoder = multiscale_se_resnext(9)
            self.img_fc = nn.Linear(2048, 256)

        elif backbone1 == 'multiscale_se_resnext_cat' :
            self.img_encoder = multiscale_se_resnext(9)
            self.img_fc = nn.Linear(1024, 256)

        elif backbone1 == 'multiscale_se_resnext_HR' :
            self.img_encoder = multiscale_se_resnext_HR(9)
            self.img_fc = nn.Linear(2048, 256)

        elif backbone1 == 'se_resnet50' :
            self.img_encoder = se_resnet50(9, None)
            print("load pretrained model from /home/zxw/2019BaiduXJTU/se_resnet50-ce0d4300.pth")
            state_dict = torch.load('/home/zxw/2019BaiduXJTU/se_resnet50-ce0d4300.pth')

            state_dict.pop('last_linear.bias')
            state_dict.pop('last_linear.weight')
            self.img_encoder.load_state_dict(state_dict, strict = False)

            self.img_fc = nn.Linear(2048, 256)

        self.dropout = nn.Dropout(0.5)
        self.cls = nn.Linear(512, 9) 

    def forward(self, x_img,x_vis):
        x_img = self.img_encoder(x_img)
        x_img = self.dropout(x_img)
        x_img = self.img_fc(x_img)

        x_vis=self.visit_model(x_vis)

        x_cat = torch.cat((x_img,x_vis),1)
        x_cat = self.cls(x_cat)
        return x_cat
