import torch
import torch.utils.data as data
import cv2
import PIL
import numpy as np
import csv
import os
import torchvision.transforms as tr
import copy as cp
import random 
from imgaug import augmenters as iaa

CLASSES = ('001', '002', '003', '004', '005', '006', '007', '008', '009')

class MM_BDXJTU2019(data.Dataset):
    def __init__(self, root, mode, transform = None):
        # mode can be valued as "train", "val", "trainval"
        self.root = root
        self.mode = mode
        with open(os.path.join(self.root, 'train', mode + '.txt')) as f:
            reader = f.readlines()
            self.file_paths = [row[:-1] for row in reader]
        self.cls_to_id = dict(zip(CLASSES, range(len(CLASSES))))
        self.normal = tr.Compose([
                           tr.ToPILImage(),
                           tr.ToTensor(),
                           tr.Normalize(mean = (0.4685483813116096, 0.538136651819416, 0.6217816988531444), std = (0.1016119525359456, 0.0900060860845122, 0.08024531900661314))
                       ])
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        item = self.file_paths[index]
        Img = cv2.imread(os.path.join(self.root, 'train', item))

        if self.mode[-5: ] == "train" or self.mode[-3: ] == 'all':
            Img = self.augumentor(Img)

        Img = cv2.resize(Img, (100, 100))        #Img = cv2.resize(Img, (331, 331))
        Img = Img[:, :, (2, 1, 0)]

        visit = self.read_npy(index)
        #visit = self.flip_week(visit)
        visit = visit.transpose(1,2,0)


        visit = tr.Compose([tr.ToTensor()])(visit)


        Img = self.normal(Img)
        Img = np.asarray(Img)
        Img = Img.astype(np.float)
        #visit = visit.astype(np.float)
        #Img = Img - self.mean
        

        Input_tensor = torch.from_numpy(Img).type(torch.FloatTensor)#.permute(2, 0, 1)

        visit_tensor = visit.float()

        Anno = self.cls_to_id[item[0:3]]
        #Input_tensor = tr.Normalize(torch.,Input_tensor)

        # return a input data like (3, 224, 224) and an interger signifies the class
        return Input_tensor, visit_tensor, Anno

    def read_npy(self,index):
        filename = self.file_paths[index][4:-4]

        pth=os.path.join(self.root, 'npy', 'train_visit' ,filename+'.npy')
        visit=np.load(pth)
        return visit

    def flip_week(self, visit, p = 0.5):
        rand = random.random()
        if rand > p:
            return visit[::-1,:,:].copy() #copy can deleminate negative  index
        else :
            return visit

    def augumentor(self,image):
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.SomeOf((0,4),[
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
            ]),
            iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
            #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            ], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

class MM_BDXJTU2019_TTA(data.Dataset):
    def __init__(self, root, mode, transform = None):
        # mode can be valued as "train", "val", "trainval"
        self.root = root
        with open(os.path.join(self.root, 'train', mode + '.txt')) as f:
            reader = f.readlines()
            self.file_paths = [row[:-1] for row in reader]
        self.cls_to_id = dict(zip(CLASSES, range(len(CLASSES))))
        self.normal = tr.Compose([
                           tr.ToPILImage(),
                           tr.ToTensor(),
                           tr.Normalize(mean = (0.4685483813116096, 0.538136651819416, 0.6217816988531444),std = (0.1016119525359456, 0.0900060860845122, 0.08024531900661314))
                       ])

        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        item = self.file_paths[index]
        Img = cv2.imread(os.path.join(self.root, 'train', item))
        Img = cv2.resize(Img, (100, 100))
        #Img = cv2.resize(Img, (100, 100))
        Img = Img[:, :, (2, 1, 0)]

        if self.transform is not None:
            Img = self.transform(Img)

        else :
            Img_O = self.normal(Img)
            Img_H = self.normal(np.fliplr(Img))
            Img_V = self.normal(np.flipud(Img))

            Img_O = np.asarray(Img_O)
            Img_H = np.asarray(Img_H)
            Img_V = np.asarray(Img_V)

        visit = self.read_npy(index)
        visit = visit.transpose(1,2,0)
        visit = tr.Compose([tr.ToTensor()])(visit)
        visit_tensor = visit.float()

        #Img = Img - self.mean
        Input_O = torch.from_numpy(Img_O.astype(np.float)).type(torch.FloatTensor)
        Input_H = torch.from_numpy(Img_H.astype(np.float)).type(torch.FloatTensor)
        Input_V = torch.from_numpy(Img_V.astype(np.float)).type(torch.FloatTensor)
        Anno = self.cls_to_id[item[0:3]]
        #Input_tensor = tr.Normalize(torch.,Input_tensor)

        # return a input data like (3, 224, 224) and an interger signifies the class
        return Input_O, Input_H, Input_V, visit_tensor,Anno


    def read_npy(self,index):
        filename = self.file_paths[index][4:-4]

        pth=os.path.join(self.root, 'npy', 'train_visit' ,filename+'.npy')
        visit=np.load(pth)
        return visit

class BDXJTU2019_test(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.ids = os.listdir(os.path.join(self.root, 'test'))
        self.ids.sort()

        self.normal = tr.Compose([
                           tr.ToPILImage(),
                           tr.ToTensor(),
                           tr.Normalize(mean = (0.46609925841978983, 0.536104764485982, 0.6208536128488661),std = (0.10149126354161417, 0.0898203268620884, 0.08024559222777607))
                       ])

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        item = self.ids[index]
        Img = cv2.imread(os.path.join(self.root, 'test', item))
        Img = cv2.resize(Img, (100, 100))
        Img = Img[:, :, (2, 1, 0)]

        Img_O = self.normal(Img)
        Img_O = np.asarray(Img_O)
        Input_O = torch.from_numpy(Img_O.astype(np.float)).type(torch.FloatTensor)

        Img_H = self.normal(np.fliplr(Img))

        Img_H = np.asarray(Img_H)
        Input_H = torch.from_numpy(Img_H.astype(np.float)).type(torch.FloatTensor)

        """  
        Img_V = self.normal(np.flipud(Img))
        Img_V = np.asarray(Img_V)

        Input_V = torch.from_numpy(Img_V.astype(np.float)).type(torch.FloatTensor)
        """
        visit = self.read_npy(index)
        visit = visit.transpose(1,2,0)
        visit = tr.Compose([tr.ToTensor()])(visit)
        visit_tensor = visit.float()

        return Input_O, Input_H, visit_tensor, item

    def read_npy(self,index):
        filename = self.ids[index][:-4]

        pth=os.path.join(self.root, 'npy', 'test_visit' ,filename + '.npy')
        visit=np.load(pth)

        return visit

class BDXJTU2019_test_MS(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.ids = os.listdir(os.path.join(self.root, 'test'))
        self.ids.sort()

        self.normal = tr.Compose([
                           tr.ToPILImage(),
                           tr.ToTensor(),
                           tr.Normalize(mean = (0.46609925841978983, 0.536104764485982, 0.6208536128488661),std = (0.10149126354161417, 0.0898203268620884, 0.08024559222777607))
                       ])

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        item = self.ids[index]
        Img = cv2.imread(os.path.join(self.root, 'test', item))
        Img = cv2.resize(Img, (331, 331))
        Img = Img[:, :, (2, 1, 0)]

        Img_O = self.normal(Img)
        Img_H = self.normal(np.fliplr(Img))
        Img_V = self.normal(np.flipud(Img))

        Img_O = np.asarray(Img_O)
        Img_H = np.asarray(Img_H)
        Img_V = np.asarray(Img_V)

        Input_O = torch.from_numpy(Img_O.astype(np.float)).type(torch.FloatTensor)
        Input_H = torch.from_numpy(Img_H.astype(np.float)).type(torch.FloatTensor)
        Input_V = torch.from_numpy(Img_V.astype(np.float)).type(torch.FloatTensor)

        return Input_O, Input_H, Input_V, item

class Augmentation(object):
    def __init__(self):
        self.augment = tr.Compose([
                           tr.ToPILImage(),
                           #tr.Pad(padding =20, fill = 0),
                           #tr.RandomAffine(degrees = 180, translate = (0, 0), shear = 20, resample = PIL.Image.BICUBIC),
                           #tr.CenterCrop(224),
                           tr.RandomHorizontalFlip(),
                           tr.RandomVerticalFlip(),
                           tr.ColorJitter(brightness=0.3, contrast=0.3, saturation = 0.3, hue = 0.1),
                           tr.ToTensor(),
                           tr.Normalize(mean = (0.4685483813116096, 0.538136651819416, 0.6217816988531444), std = (0.1016119525359456, 0.0900060860845122, 0.08024531900661314))
                       ])
    def __call__(self, Img):
        Img = self.augment(Img)
        return np.asarray(Img)


        
        
