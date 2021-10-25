"""
create the dataset
"""
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

import pandas as pd
import numpy as np
import read_data
import os
import shutil
from PIL import Image

def move_image():
    '''
    move image files to k-fold folder
    '''

    df = read_data.load_dataframe()
    
    # create directory
    pre_path = './data/processed'
    for i in range(5):
        path = os.path.join(pre_path, str(i))
        if not os.path.exists(path):
            os.mkdir(path)
            print('made directorary ', path)
        
    # move images to correpsponding folder and rename it
    identities = df[['fold', 'class', 'image_path', 'name']].to_numpy()
    for identity in identities:
        original_path = str(identity[2])
        # to path = pre_path/fold/name+class.png
        to_path = os.path.join(pre_path, str(identity[0]), identity[3]+f'_{identity[1]}.png')
        shutil.copy(original_path, to_path)


def get_images(df):
    # l = path, fold
    l = df[['image_path', 'fold']].to_numpy()
    image_fold = []
    for path, fold in l:
        im = Image.open(path)
        image_fold.append([im, fold])
    return image_fold


def get_dataset(k):
    '''
    split the dataset training/test set
    '''
    training = list(range(5))
    training.remove(k)
    test = [k]

    df = read_data.load_dataframe()
    training_list = df[df['fold'].isin(training)][['image_path', 'class']].to_numpy()
    test_list = df[df['fold'].isin(test)][['image_path', 'class']].to_numpy()

    dataset_train = ImageSet(training_list, 'train')
    dataset_test = ImageSet(test_list, 'test')

    return dataset_train, dataset_test
    

def get_crop_dataset(k):
    '''
    split the crop images dataset training/test set
    '''
    training = list(range(5))
    training.remove(k)
    test = [k]

    df = read_data.load_dataframe()
    training_list = df[df['fold'].isin(training)][['crop_path', 'class']].to_numpy()
    test_list = df[df['fold'].isin(test)][['crop_path', 'class']].to_numpy()

    dataset_train = ImageSet(training_list, 'train')
    dataset_test = ImageSet(test_list, 'test')

    return dataset_train, dataset_test
    
def get_original_dataset(k):
    '''
    split the crop images dataset training/test set
    '''
    training = list(range(5))
    training.remove(k)
    test = [k]

    df = read_data.load_dataframe()
    training_list = df[df['fold'].isin(training)][['image_path', 'class']].to_numpy()
    test_list = df[df['fold'].isin(test)][['image_path', 'class']].to_numpy()

    transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
    
    dataset_train = OringalSet(training_list, transform)
    dataset_test = OringalSet(test_list, transform)

    return dataset_train, dataset_test


class ImageSet(torch.utils.data.Dataset):
    def __init__(self, image_list, type='train') -> None:
        self.image_list = image_list
        self.type = type

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        self.data_transforms = {
            'train':
            transforms.Compose([
                transforms.Resize((224,224)),
                #torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)
                # degree: rotation, translate: pingyi(chinese), scale, shear 则仅在 x 轴错切，在 (-a, a) 之间随机选择错切角度
                transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'test':
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize
            ]),
        }
    
    def __getitem__(self, index):
        image_path = self.image_list[index][0]
        label = self.image_list[index][1]
        img = Image.open(image_path)
        data = self.data_transforms[self.type](img)
        return [data, label]

    def __len__(self):
        return self.image_list.shape[0]



class OringalSet(torch.utils.data.Dataset):
    def __init__(self, image_list, transform) -> None:
        self.image_list = image_list
        self.type = type
        self.trainsform = transform
          
    def __getitem__(self, index):
        image_path = self.image_list[index][0]
        label = self.image_list[index][1]
        img = Image.open(image_path)
        data = self.trainsform(img)
        return [data, label]

    def __len__(self):
        return self.image_list.shape[0]
            





    