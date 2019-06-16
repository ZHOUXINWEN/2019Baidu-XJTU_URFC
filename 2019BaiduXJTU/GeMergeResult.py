import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from DataLoader import TiangongResultMerge
from basenet.ResNeXt101_64x4d import ResNeXt101_64x4d
from basenet.pnasnet import pnasnet5large
from basenet.senet import se_resnext101_32x4d,se_resnet101,se_resnet50
import os

CLASSES = ('DESERT', 'MOUNTAIN', 'OCEAN', 'FARMLAND', 'LAKE', 'CITY')

def GeResult():
    # Priors
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    # Dataset
    Dataset = TiangongResultMerge(root = 'data')
    Dataloader = data.DataLoader(Dataset, 1,
                                 num_workers = 1,
                                 shuffle = True, pin_memory = True)

    # Network
    
    #Network = pnasnet5large(6, None)
    Network = ResNeXt101_64x4d(6)
    net = torch.nn.DataParallel(Network, device_ids=[0])
    cudnn.benchmark = True
    
    Network.load_state_dict(torch.load('weights/aug_ResNeXt/_Tiangong_SGD_85.pth'))
    #net = torch.load('weights/aug_ResNeXt/Tiangong55000use.pth')
    Network.eval()



    Network2 = pnasnet5large(6, None)
    net2 = torch.nn.DataParallel(Network2, device_ids=[0])
    cudnn.benchmark = True
    
    Network2.load_state_dict(torch.load('weights/aug_fix1block_pnasnet/_Tiangong_SGD_85.pth'))
    net2.eval()

    Network3 = se_resnet50(6, None)
    net3 = torch.nn.DataParallel(Network3, device_ids=[0])
    cudnn.benchmark = True
    
    Network3.load_state_dict(torch.load('weights/aug_se_resnet50/_Tiangong_SGD_95.pth'))
    #Network3.load_state_dict(torch.load('weights/ResSample_aug_se_resnet50/_Tiangong_SGD_60.pth'))
    net3.eval()


    filename = 'Rejection_se_resnet50_pnasnet_resnext.csv'
    # Result file preparation
    if os.path.exists(filename):
        os.remove(filename)
    os.mknod(filename)

    f = open(filename, 'w')

    for (imgs,img2,anos) in Dataloader:
        imgs = imgs.cuda()
        pred1 = Network.forward(imgs)
        pred2 = net2.forward(img2)
        pred3 = net3.forward(imgs)
        # eliminate the predictions that has low probality
        '''

        
        
        pred1 = torch.nn.functional.normalize(pred1)
        pred2 = torch.nn.functional.normalize(pred2)
        pred3 = torch.nn.functional.normalize(pred3)
        '''
        pred1 = pred1 + 2*torch.mul(pred1, torch.le(pred1,0).float())    
        pred2 = pred2 + 2*torch.mul(pred2, torch.le(pred2,0).float())
        pred3 = pred3 + 2*torch.mul(pred3, torch.le(pred3,0).float())
        #print(pred1)
        #pred1 = torch.mul(pred1, torch.ge(pred1,torch.topk(pred1, dim = 1, k=3, largest = True)[0][2]))   
        #pred2 = torch.mul(pred2, torch.ge(pred2,torch.topk(pred2, k=3, largest = True)[0][2]))
        #pred3 = torch.mul(pred3, torch.ge(pred3,torch.topk(pred3, k=3, largest = True)[0][2]))
      
        preds = torch.add(pred1, pred2)
        preds.add(pred3)
        _, pred = preds.data.topk(1, 1, True, True)
        f.write(anos[0] + ',' + CLASSES[pred[0][0]] + '\r\n')

if __name__ == '__main__':
    GeResult()
        
