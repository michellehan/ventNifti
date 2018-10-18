from __future__ import division

import os
import warnings
import pandas as pd
import numpy as np
import random
import nibabel

import torch
import torchvision.transforms as transforms
#import torch.utils.transforms as extended_transforms
from torch.utils.data import Dataset, DataLoader

from . import data
from .utils import export


from skimage import io
from PIL import Image
from sklearn.metrics import roc_auc_score
from skimage.transform import resize

######################################################
######################################################
######################################################

@export
def cxr14():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]
                                     )

    train_transformation = data.TransformTwice(transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize,
                                                ]))

    eval_transformation = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                        ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        # 'datadir': '../data/cxr14/',
        # 'csvdir': '../data_csv/',
        # 'num_classes': None
    }


class MaskToTensor(object):
    def __call__(self, img):
            return torch.from_numpy(np.array(img, dtype=np.int32)).long()

def RotateFlip(angle, flip): 
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = transforms.Compose([
            transforms.RandomRotation(degrees=(angle,angle)),
            transforms.RandomHorizontalFlip(p=flip),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
    target_transformation = transforms.Compose([
            transforms.RandomRotation(degrees=(angle,angle)),
            transforms.RandomHorizontalFlip(p=flip),
            transforms.Resize(256),
            transforms.ToTensor()
    ])

    return train_transformation, target_transformation


def RotateFlipFlip(angle, hflip, vflip): 
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
 
    train_transformation = transforms.Compose([
    #        transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(angle,angle)),
            transforms.RandomHorizontalFlip(p=hflip),
            transforms.RandomVerticalFlip(p=vflip),
            transforms.Resize(256),
            transforms.ToTensor(),
#            transforms.Normalize(**channel_stats)
        ])
    target_transformation = transforms.Compose([
    #        transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(angle,angle)),
            transforms.RandomHorizontalFlip(p=hflip),
            transforms.RandomVerticalFlip(p=vflip),
            transforms.Resize(256),
            transforms.ToTensor()
    ])

    return train_transformation, target_transformation



@export
def ventricleNormal():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

    chance = random.random()
    angles = range(-5,6) #rotate angles -5 to 5
    num_transforms = len(angles)

    #for i in range(num_transforms * 2):
    #    if i/(num_transforms * 2) <= chance < (1 + i)/(num_transforms * 2):
    #        train_transformation, target_transformation = RotateFlip( angles[i % num_transforms], i // num_transforms)
    for i in range(num_transforms * 4):
        if i/(num_transforms * 4) <= chance < (1 + i)/(num_transforms * 4):
            train_transformation, target_transformation = RotateFlipFlip( angles[i % num_transforms], i // num_transforms, (i // num_transforms) % 2)

    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
 #       transforms.Normalize(**channel_stats)
    ])

    eval_target_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    return {
        'train_transformation': train_transformation,
        'target_transformation': target_transformation,
        'eval_transformation': eval_transformation,
        'eval_target_transformation': eval_target_transformation
    }

@export
def imagenet():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation
        #'datadir': 'data-local/images/ilsvrc2012/',
        #'num_classes': 1000
    }


@export
def cifar10():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/cifar/cifar10/by-image',
        'num_classes': 10
    }


### complete version similar to torchvision.datasets.ImageFolder / torchvision.datasets.DatasetFolder
class ChestXRayDataset(Dataset):
    """ CXR8 dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        df = pd.read_csv(csv_file)
        
        classes = df.columns[3:].values.tolist()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.idx_to_class = dict(enumerate(classes))
        self.classes = classes
        
        samples = []
        for idx in range(len(df)):
            path = df.iloc[idx]['image_path']
            target = df.iloc[idx, 3:].as_matrix().astype('float32')       ### labels type: array
            item = (path, target)
            samples.append(item)
        assert(len(samples) == len(df))
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
         
        path, target = self.samples[index]
        
        ### load image
        img_name = os.path.join(self.root_dir, path)       ### get 'image_path'
        image = io.imread(img_name)
        if(len(image.shape) == 3):                          ### some samples have four channels
            image = image[:,:,0]
        h, w = image.shape
        c = 3
        images = np.zeros((h, w, c), dtype = np.uint8)      ### Set image channel dim = 3
        for i in range(c):
            images[:,:,i] = image 
        assert(images.shape == (1024,1024,3))
        images = Image.fromarray(images)

        if self.transform:
            images = self.transform(images)

        ### load labels
        labels = torch.from_numpy(target)
  
        ### return tuple
        return (images, labels)



class IVCdataset(Dataset):
    def __init__(self, csv_file, path, transform=None):
        """
        csv_file = csv where first column = image filenames and second column = classification
        path = directory to all iamges
        """
        self.path = path
        self.transform = transform

        df = pd.read_csv(csv_file, header=None)

        classes = df.iloc[:,1].values.tolist()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.idx_to_class = dict(enumerate(classes))
        self.classes = classes
        print("> dataset size: ", df.shape[0])

        #load labels
        samples = []
        for i in range(len(df)):
            name = df.iloc[i,0]
            target = df.iloc[i,1].astype('int_')
            item = (name, target)
            samples.append(item)
        assert(len(samples) == len(df))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img_name = os.path.join(self.path, path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = io.imread(img_name)

        if (len(image.shape)==3):
            image = image[:,:,0]

        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        #    image = resize(image, (224,224))
        image = image.astype('float32') 
        
        h, w  = image.shape
        c = 3
        images = np.zeros((h, w, c), dtype = np.uint8)
        for i in range(c):
            images[:,:,i] = image
        #assert(images.shape == (1024,1024,3))

        images = Image.fromarray(images)         

        #trans = transforms.ToTensor()
        #images = trans(images) 
        
        if self.transform:
            images = self.transform(images)
        
        labels = torch.from_numpy(np.array([target]))
        return (images, labels)


def loadImages(image, basedir):
    #img_name = os.path.join(basedir, image)
 
    #img_name = nibabel.load(img_name).get_data()

    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    #    image = io.imread(img_name)
    if (len(image.shape)==3):
        image = image[:,:,0]
    image = image.astype('float32') 
    h, w  = image.shape
    c = 3
    images = np.zeros((h, w, c), dtype = np.uint8)
    for i in range(c):
        images[:,:,i] = image
    images = Image.fromarray(images)         
    return images


class Ventricles(Dataset):
    def __init__(self, csv_file, path_raw, path_segs, input_transform=None, target_transform=None, train=False):
        self.path_raw = path_raw
        self.path_segs = path_segs
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.train = train

        df = pd.read_csv(csv_file, header=None)
        #print("Dataset size: ", df.shape[0])

        samples = []

        #lower = round( len(df) / 5 )
        #upper = round( len(df) / 5 * 4 )
        for i in range(len(df)):
            name = df.iloc[i,0]
            target = df.iloc[i,1]

            image_name = os.path.join(path_raw, name)
            target_name = os.path.join(path_segs, target)
            image_ni = nibabel.load(image_name).get_data()
            target_ni = nibabel.load(target_name).get_data()


            slices = image_ni.shape[2]
            lower = slices / 4
            upper = slices / 4 * 3
            for i in range(slices):
                name = image_ni[:,:,i]
                target = target_ni[:,:,i]
                item = (name, target)
                samples.append(item)
                
                if train and lower < i < upper:
                    for _ in range(3): samples.append(item)
        self.samples = samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        #images, targets = self.samples[index]
        image, target = self.samples[index]
        images = loadImages(image, self.path_raw)
        targets = loadImages(target, self.path_segs)
        #images = image
        #targets = target
        tobinary = targets.convert('L')
        targets_mask = tobinary.point(lambda x: 0 if x < 1 else 1, '1')



        if self.train:
            channel_stats = dict(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

            chance = random.random()
            angle = range(-5,6) #rotate angles -5 to 5
            n_angles = len(angle)

            #for i in range(num_transforms * 2):
            #    if i/(num_transforms * 2) <= chance < (1 + i)/(num_transforms * 2):
            #        train_transformation, target_transformation = RotateFlip( angles[i % num_transforms], i // num_transforms)
            #        print('angles/flip', angles[i % num_transforms], i // num_transforms)    
            for i in range(n_angles * 4):
                if i/(n_angles * 4) <= chance < (1 + i)/(n_angles * 4):
                    input_transform, target_transform = RotateFlipFlip( angle[i % n_angles], i // n_angles, (i // n_angles) % 2)

            images = input_transform(images)
            targets_mask = target_transform(targets_mask)
            #targets_mask = target_transform(targets)

        else:
            if self.input_transform:
                images = self.input_transform(images)
            if self.target_transform:
                targets_mask = self.target_transform(targets_mask)

        return (images, targets_mask)


