import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2
from utils import readlines, pil_loader


class CHAOS(data.Dataset):
    def __init__(self, data_path, istrain):
        self.datapath = data_path
        self.istrain = istrain
        self.loader = pil_loader
       # fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        
    def __getitem__(self, index):
        if self.istrain==True:
            txtname = train_split
            
            txtpath = os.path.join(os.path.dirname(__file__), "data", txt_name, ".txt")
            
            filenames = readline(path)
            line = filenames[index]
            
            inputs = {}
            inputs{"InPhase"} = self.get_color(self.datapath, 'in', line)
            inputs{"OutPhase"} = self.get_color(self.datapath, 'out', line)
            inputs{"gt"} = self.get_color(self.datapath, 'gt', line)
            
            
            
        else:
            txtname=test_split
            txtpath=os.path.join(os.path.dirname(__file__), "data", txt_name, ".txt")
            filename=readline(txt_name)
            inputs={}
            
            
        return inputs
    def __len__(self, ):
        
        
        
        
    def get_color(self, data_path, img_type, img_name, do_flip):
        color = self.loader(os.path.join(data_path, img_type, img_name))
#         if do_flip:
#             color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color


class SkinDataset(data.Dataset):
    """
    dataloader for skin lesion segmentation tasks
    """
    def __init__(self, image_root, gt_root):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        
        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ]
        )

    def __getitem__(self, index):
        
        image = self.images[index]
        gt = self.gts[index]
        gt = gt/255.0

        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        gt = self.gt_transform(transformed['mask'])
        return image, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = SkinDataset(image_root, gt_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)
        gt = self.gts[self.index]
        gt = gt/255.0
        self.index += 1

        return image, gt



if __name__ == '__main__':
    path = 'data/'
    tt = SkinDataset(path+'data_train.npy', path+'mask_train.npy')

    for i in range(50):
        img, gt = tt.__getitem__(i)

        img = torch.transpose(img, 0, 1)
        img = torch.transpose(img, 1, 2)
        img = img.numpy()
        gt = gt.numpy()

        plt.imshow(img)
        plt.savefig('vis/'+str(i)+".jpg")
 
        plt.imshow(gt[0])
        plt.savefig('vis/'+str(i)+'_gt.jpg')
