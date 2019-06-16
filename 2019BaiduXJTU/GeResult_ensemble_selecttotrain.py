import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from Dataloader.MultiModal_BDXJTU2019 import BDXJTU2019_test
from basenet.ResNeXt101_64x4d import ResNeXt101_64x4d
from basenet.senet import se_resnet50,se_resnext101_32x4d
from basenet.octave_resnet import octave_resnet50
from basenet.nasnet import nasnetalarge
from basenet.multimodal import MultiModalNet

import os

CLASSES = ['001', '002', '003', '004', '005', '006', '007', '008', '009']

def GeResult():
    # Priors
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    # Dataset
    Dataset = BDXJTU2019_test(root = 'data')
    Dataloader = data.DataLoader(Dataset, 1,
                                 num_workers = 1,
                                 shuffle = False, pin_memory = True)

    # Network
    cudnn.benchmark = True    
    #Network = pnasnet5large(6, None)
    #Network = ResNeXt101_64x4d(6)
    net1 =MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)
    net1.load_state_dict(torch.load('/home/zxw/2019BaiduXJTU/weights/MultiModal_se_resnext50_32x4d_resample_pretrained/BDXJTU2019_SGD_16.pth'))
    net1.eval()

    net2 = MultiModalNet('multiscale_se_resnext_HR', 'DPN26', 0.5)
    net2.load_state_dict(torch.load('/home/zxw/2019BaiduXJTU/weights/MultiModal_50_MS_resample_pretrained_HR/BDXJTU2019_SGD_26.pth'))
    net2.eval()


    net3 = MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)
    net3.load_state_dict(torch.load('/home/zxw/2019BaiduXJTU/weights/MultiModal_se_resnext50_32x4d_resample_pretrained_w/BDXJTU2019_SGD_50.pth'))
    net3.eval()


    net4 =MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)
    net4.load_state_dict(torch.load('/home/zxw/2019BaiduXJTU/weights/MultiModal_se_resnext50_32x4d_resample_pretrained_1/BDXJTU2019_SGD_80.pth'))
    net4.eval()

    filename = 'MM_epoch26_25_all_pretrained_2HR_616v2.txt'

    f = open(filename, 'w')

    for (Input_O, Input_H, visit_tensor, anos) in Dataloader:
        ConfTensor_O = net1.forward(Input_O.cuda(), visit_tensor.cuda())
        ConfTensor_H = net2.forward(Input_O.cuda(), visit_tensor.cuda())
        ConfTensor_V = net3.forward(Input_O.cuda(), visit_tensor.cuda())
        ConfTensor_1 = net4.forward(Input_O.cuda(), visit_tensor.cuda())
        preds = torch.nn.functional.normalize(ConfTensor_O) + torch.nn.functional.normalize(ConfTensor_H) +torch.nn.functional.normalize(ConfTensor_V) +torch.nn.functional.normalize(ConfTensor_1)
        _, pred = preds.data.topk(1, 1, True, True)
        #f.write(anos[0] + ',' + CLASSES[4] + '\r\n')
        print(anos[0][:-4] + '\t' + CLASSES[pred[0][0]] + '\n')
        f.writelines(anos[0][:-4] + '\t' + CLASSES[pred[0][0]] + '\n')
    f.close()
if __name__ == '__main__':
    GeResult()
        
