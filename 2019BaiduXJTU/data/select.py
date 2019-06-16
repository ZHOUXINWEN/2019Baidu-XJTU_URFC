import torch
import torch.utils.data as data
import cv2
import PIL
import numpy as np
import csv
import os
import torchvision.transforms as tr
import copy as cp

def write_to_txt(name, input_list):
    with open('train/cleaned_'+name+ '.txt', 'w') as f:
        for file_path in input_list:
            f.writelines(file_path+'\n')


def find_bad_sample(mode):
   
    file_paths = None
    bad_sample_list = []
    with open(os.path.join('/home/zxw/2019BaiduXJTU/data', 'train', mode + '.txt')) as f:
        reader = f.readlines()
        file_paths = [row[:-1] for row in reader]
    
    final_list = cp.deepcopy(file_paths)

    for i in file_paths:
        ratio = judge_bad_sample(i)

        if ratio> 0.25 :
            bad_sample_list.append(i)
            os.system('cp /home/zxw/2019BaiduXJTU/data/train/'+ i +' /home/zxw/2019BaiduXJTU/data/bad_sample')
            final_list.remove(i)
    
    print(len(final_list))
    return final_list

def judge_bad_sample(img_name):
    Img = cv2.imread(os.path.join('/home/zxw/2019BaiduXJTU/data', 'train', img_name))
    Gray = tr.Compose([
                       tr.ToPILImage(),
                       tr.Grayscale(),
                       tr.ToTensor()
                       ])

    Img_Gray = Gray(Img)
    ratio = float(torch.eq(Img_Gray, 0.0).sum())/10000.0

    #print(ratio) 
    #print(img_name, float(torch.eq(Img_Gray, 0.0).sum() + torch.eq(Img_Gray, 1.0).sum())/10000.0)
    return ratio

def judge_bad_sample_test(img_name):
    Img = cv2.imread(os.path.join('/home/zxw/2019BaiduXJTU/data/', img_name))
    Gray = tr.Compose([
                       tr.ToPILImage(),
                       tr.Grayscale(),
                       tr.ToTensor()
                       ])

    Img_Gray = Gray(Img)
    ratio = float(torch.eq(Img_Gray, 0.0).sum())/10000.0

    #print(ratio) 
    #print(img_name, float(torch.eq(Img_Gray, 0.0).sum() + torch.eq(Img_Gray, 1.0).sum())/10000.0)
    return ratio

if __name__ == '__main__':
    mode = 'all'
    #judge_bad_sample('008/013186_008.jpg')
    #fl = find_bad_sample(mode)
    #write_to_txt(mode, fl)

    with open(os.path.join('/home/zxw/2019BaiduXJTU/data/',  'test.txt')) as f:
        reader = f.readlines()
       
        file_paths = [row[:-1] for row in reader]
        final_list = cp.deepcopy(file_paths)
        for i in file_paths:
            ratio = judge_bad_sample_test(i)

            if ratio> 0.25 :
                #bad_sample_list.append(i)
                #os.system('cp /home/zxw/2019BaiduXJTU/data/train/'+ i +' /home/zxw/2019BaiduXJTU/data/bad_sample')
                final_list.remove(i)
        write_to_txt('test', final_list)
        print(len(final_list))













