import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import math

from Dataloader.MultiModal_BDXJTU2019 import MM_BDXJTU2019, Augmentation
from basenet.ResNeXt101_64x4d import ResNeXt101_64x4d
from basenet.senet import se_resnet50,se_resnext101_32x4d
from basenet.oct_resnet import oct_resnet26,oct_resnet101
from basenet.nasnet import nasnetalarge
from basenet.multiscale_resnet import multiscale_resnet
from basenet.multimodal import MultiModalNet
from basenet.multiscale_se_resnext import multiscale_se_resnext
from torch.utils.data.sampler import WeightedRandomSampler

import argparse

parser = argparse.ArgumentParser(description = 'BDXJTU')
parser.add_argument('--dataset_root', default = 'data', type = str)
parser.add_argument('--class_num', default = 9, type = int)
parser.add_argument('--batch_size', default =64, type = int)
parser.add_argument('--num_workers', default = 4, type = int)
parser.add_argument('--start_iter', default = 0, type = int)
parser.add_argument('--adjust_iter', default = 40000, type = int)
parser.add_argument('--end_iter', default = 60000, type = int)
parser.add_argument('--lr', default = 0.01, type = float)
parser.add_argument('--momentum', default = 0.9, type = float)
parser.add_argument('--weight_decay', default = 1e-5, type = float)
parser.add_argument('--gamma', default = 0.1, type = float)
parser.add_argument('--resume', default = None, type = str)
parser.add_argument('--basenet', default = 'se_resnext101_32x4d', type = str)
parser.add_argument('--print-freq', '-p', default=20, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

#parser.add_argument('--fixblocks', default = 2, type = int)

args = parser.parse_args()
class_num = [9542, 7538, 3590, 1358, 3464, 5507, 3517, 2617, 2867]
class_ration = [40000.0/i for i in class_num]

diag_prec = [0.76765499, 0.68981794, 0.6128591, 0.58947368, 0.90697674, 0.58221024, 0.6407767,  0.54887218, 0.61148649]
#[0.76765499, 0.68981794, 0.6128591, 0.58947368, 0.90697674, 0.58221024, 0.6407767,  0.54887218, 0.61148649]
#[0.76765499, 0.68981794, 0.6128591, 0.58947368, 0.90697674, 0.58221024, 0.6407767,  0.54887218, 0.61148649] _1

MAX = max(diag_prec)
weights = [MAX/i for i in diag_prec]
weights = torch.tensor(weights)#torch.nn.functional.normalize(torch.tensor([2.0, 3.0, 4.0, 4.0, 1.0, 4.0, 4.0, 5.0, 3.0]))

def main():
    #create model
    best_prec1 = 0

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    if args.basenet == 'MultiModal':
        model = MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)    
        #net = Networktorch.nn.DataParallel(Network, device_ids=[0])


    elif  args.basenet == 'oct_resnet101':
        model = oct_resnet101()    
        #net = Networktorch.nn.DataParallel(Network, device_ids=[0])


    model = model.cuda()
    cudnn.benchmark = True

    # Dataset
    Aug = Augmentation()
    Dataset_train = MM_BDXJTU2019(root = args.dataset_root, mode = 'MM_1_train', transform = Aug)
    #weights = [class_ration[label] for data,label in Dataset_train]

    Dataloader_train = data.DataLoader(Dataset_train, args.batch_size, 
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)

    Dataset_val = BDXJTU2019(root = args.dataset_root, mode = 'val')
    Dataloader_val = data.DataLoader(Dataset_val, batch_size = 8,
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)

    criterion = nn.CrossEntropyLoss(weight = weights).cuda()

    Optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, momentum = args.momentum,
                          weight_decay = args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(Optimizer, epoch)

        # train for one epoch
        train(Dataloader_train, model, criterion, Optimizer, epoch)    #train(Dataloader_train, Network, criterion, Optimizer, epoch)

        # evaluate on validation set
        #_,_ = validate(Dataloader_val, model, criterion)  #prec1 = validate(Dataloader_val, Network, criterion)

        # remember best prec@1 and save checkpoint
        #is_best = prec1 > best_prec1
        #best_prec1 = max(prec1, best_prec1)
        #if is_best:
        if epoch%1 == 0:
            torch.save(model.state_dict(), 'weights/'+ args.basenet +'_se_resnext50_32x4d_resample_pretrained_80w_1/'+ 'BDXJTU2019_SGD_' + repr(epoch) + '.pth')

        

def train(Dataloader,model, criterion, optimizer, epoch):
    # Priors

    # Dataset
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
        
    model.train()
    model = model.cuda()
    DatasetLen = len(Dataloader)
    warmup_list = [0,1]
    warmup_len = DatasetLen*len(warmup_list)

    #cl = nn.CrossEntropyLoss()
    # Optimizer

    #Optimizer = optim.RMSprop(net.parameters(), lr = args.lr, momentum = args.momentum,
                          #weight_decay = args.weight_decay)

    # train
    end = time.time()
    for i, (input_img, input_vis, anos) in enumerate(Dataloader):
        data_time.update(time.time() - end)

        target = anos.cuda(async=True)
        
        if epoch in warmup_list:
            for param_group in optimizer.param_groups:
                cur_iter = float(i + 1 + DatasetLen*epoch)
                param_group['lr'] = args.lr*(cur_iter/warmup_len)
        
        with torch.no_grad():
            input_img_var = Variable(input_img.cuda())
            input_vis_var = Variable(input_vis.cuda())
            target_var = Variable(target.cuda())

        # compute output
        output = model(input_img_var.cuda(), input_vis_var.cuda())
        #print(target_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input_img.size(0))
        top1.update(prec1[0], input_img.size(0))
        top5.update(prec5[0], input_img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % 200 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i+1, len(Dataloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))        
            losses.reset()
            top1.reset()
            top5.reset()
        torch.cuda.empty_cache()

def validate(val_loader,model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        with torch.no_grad():
            input_var = Variable(input.cuda())
            target_var = Variable(target.cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % 100 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i+1, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
        torch.cuda.empty_cache()
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (math.sqrt(0.9) ** (epoch)) ##origin 25 epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
    
