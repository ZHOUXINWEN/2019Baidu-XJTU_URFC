import torch
import torch.utils.data as data
import cv2
import PIL
import numpy as np
import csv
import os
import torchvision.transforms as tr

CLASSES = ('DESERT', 'MOUNTAIN', 'OCEAN', 'FARMLAND', 'LAKE', 'CITY')

class Tiangong(data.Dataset):
    def __init__(self, root, mode, transform = None):
        # mode can be valued as "train", "val", "trainval"
        self.root = root
        with open(os.path.join(self.root, mode + '.csv')) as f:
            reader = csv.reader(f)
            self.ids = [row for row in reader]
        self.cls_to_id = dict(zip(CLASSES, range(len(CLASSES))))
        self.mean = np.array([113.4757, 112.0985, 102.8271])
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item = self.ids[index]
        Img = cv2.imread(os.path.join(self.root, 'train', item[0]))
        Img = cv2.resize(Img, (224, 224))
        Img = Img[:, :, (2, 1, 0)]
        if self.transform is not None:
            Img = self.transform(Img)
        Img = Img.astype(np.float)
        Img = Img - self.mean
        
        Anno = self.cls_to_id[item[1]]
        return torch.from_numpy(Img).permute(2, 0, 1).type(torch.FloatTensor), Anno

class Augmentation(object):
    def __init__(self):
        self.augment = tr.Compose([
                           tr.ToPILImage(),
                           tr.Pad(100, padding_mode = 'reflect'),
                           tr.RandomAffine(degrees = 180, translate = (0, 0), shear = 20, resample = PIL.Image.BICUBIC),
                           tr.CenterCrop(224),
                           tr.ColorJitter(brightness=0.3, contrast=0.3, saturation = 0.3, hue = 0.1)
                       ])
    def __call__(self, Img):
        Img = self.augment(Img)
        return np.asarray(Img)





        
        
