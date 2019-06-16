import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import math
from basenet.senet import se_resnet50, se_resnext101_32x4d, se_resnext50_32x4d

model = se_resnext50_32x4d(9, None)    

state_dict = torch.load('se_resnext50_32x4d-a260b3a4.pth')
for k, v in state_dict.items() :
    print(k)
#print(state_dict)

state_dict.pop('last_linear.bias')
state_dict.pop('last_linear.weight')
model.load_state_dict(state_dict, strict = False)

print(model(torch.randn(16,3,100,100).float()).size())
"""
init.xavier_uniform_(model.last_linear.weight.data)
model.last_linear.bias.data.zero_()
"""
