'''
This python file is to test a single network on dataset and visiualise confusion metrix
'''
import os
import csv
import h5py
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix, accuracy_score
from Dataloader.BDXJTU2019 import BDXJTU2019, BDXJTU2019_test,BDXJTU2019_TTA
from Dataloader.MultiModal_BDXJTU2019 import MM_BDXJTU2019
from basenet.ResNeXt101_64x4d import ResNeXt101_64x4d
from basenet.senet import se_resnet50,se_resnext101_32x4d
from basenet.octave_resnet import octave_resnet50
from basenet.nasnet import nasnetalarge
from basenet.multimodal import MultiModalNet

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    print(np.diag(cm))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def GeResult():

    # Dataset
    Dataset_val = MM_BDXJTU2019(root = 'data', mode = '1_val')
    Dataloader_val = data.DataLoader(Dataset_val, batch_size = 1,
                                 num_workers = 2,
                                 shuffle = True, pin_memory = True)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    class_names = ['001', '002', '003', '004', '005', '006', '007', '008', '009']
    

    # construct network
    epoch = 80
    net = MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)
    net.load_state_dict(torch.load('/home/zxw/2019BaiduXJTU/weights/MultiModal_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_' + str(epoch) + '.pth'))
    print('load '+ str(epoch) +' epoch model')
    net.eval()

    results = []
    results_anno = []
    
    for i, (Input_img, Input_vis, Anno) in enumerate(Dataloader_val):
        Input_img = Input_img.cuda()
        Input_vis = Input_vis.cuda()


        ConfTensor = net.forward(Input_img, Input_vis)
        _, pred = ConfTensor.data.topk(1, 1, True, False)

        results.append(pred.item())

        results_anno.append(Anno)                                #append annotation results
        if((i+1) %1000 == 0 ):
            print(i+1)
            print('Accuracy of Orignal Input: %0.6f'%(accuracy_score(results, results_anno, normalize = True)))

    # print accuracy of different input
    print('Accuracy of Orignal Input: %0.6f'%(accuracy_score(results, results_anno, normalize = True)))

    cnf_matrix = confusion_matrix(results_anno, results)
    cnf_tr = np.trace(cnf_matrix)

    cnf_tr = cnf_tr.astype('float')
    print(cnf_tr/len(Dataset_val))
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = class_names ,title='Confusion matrix, without normalization')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = class_names, normalize=True, title='Normalized confusion matrix')
    plt.show()
    
if __name__ == '__main__':
    GeResult()
        
