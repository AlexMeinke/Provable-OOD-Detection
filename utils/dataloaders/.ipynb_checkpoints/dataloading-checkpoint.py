import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

import paths_config
import numpy as np
import warnings

from .tiny_utils.tinyimages_80mn_loader import TinyImages
import utils.dataloaders.noisefunctions as noisefunctions
from .auto_augment import AutoAugment
import utils.dataloaders.imagenet_subsets as imgnet
import utils.dataloaders.flowers as flowers
import utils.dataloaders.fgvc as fgvc
import utils.dataloaders.stanford_cars as cars
import os
import natsort
from PIL import Image
import random


class Noise_Dataset(torch.utils.data.dataset.Dataset):
    """A dataset that is built from a ground dataset and a noise function, returning noisy images of the same shape as the ground data.
       noise_fn should be a function accepting a ground data sample with its label and returning the noisy sample and label with the same shape as the input
    """
    
    def __init__(self, ground_ds, noise_fn, label_fn=None, transform=None):
        self.ground_ds = ground_ds
        self.noise_fn = noise_fn
        self.label_fn = label_fn
        self.transform = transform
        #self.__name__ = noise.__name__ + '_on_' + ground_ds.__name__

    def __getitem__(self, index):
        noisy = self.noise_fn(self.ground_ds[index])
        inp = noisy[0]
        if self.transform is not None:
            inp = transforms.ToPILImage()(inp)
            inp = self.transform(inp)
            
        if self.label_fn == None:
            lbl = noisy[1]
        else:
            lbl = self.label_fn(noisy[1])
        return inp, lbl

    def __len__(self):
        return len(self.ground_ds)
    
    _repr_indent = 4
    def __repr__(self):
        head = "Dataset " + self.__class__.__name__ + '_' + self.noise_fn.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)


def getloader_MNIST(train, batch_size, augmentation, dataloader_kwargs):
    if list(augmentation) == []:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif list(augmentation) == ['crop']:
        transform = transforms.Compose([
            transforms.RandomCrop(28, augmentation['crop']),
            transforms.ToTensor(),
        ])
    else:
        raise KeyError(f'Only crop augmentation supported for MNIST. Got {augmentation}.')
    dset = datasets.MNIST(paths_config.location_dict['MNIST'], train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

def getloader_FashionMNIST(train, batch_size, augmentation, dataloader_kwargs):
    if list(augmentation) == []:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif list(augmentation) == ['crop']:
        transform = transforms.Compose([
            transforms.RandomCrop(28, augmentation['crop']),
            transforms.ToTensor(),
        ])
    else:
        raise KeyError(f'Only crop augmentation supported. Got {augmentation}.')
        
    dset = datasets.FashionMNIST(paths_config.location_dict['FashionMNIST'], train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

def getloader_EMNIST_Letters(train, batch_size, augmentation, dataloader_kwargs):
    if list(augmentation) == []:
        transform = transforms.Compose([
            lambda x: np.array(x).T,
            transforms.ToTensor(),
        ])
    elif list(augmentation) == ['crop']:
        transform = transforms.Compose([
            lambda x: np.array(x).T,
            transforms.RandomCrop(28, augmentation['crop']),
            transforms.ToTensor(),
        ])
    else:
        raise KeyError(f'Only crop augmentation supported. Got {augmentation}.')
        
    dset = datasets.EMNIST(paths_config.location_dict['EMNIST'], split='letters', train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

def getloader_Omniglot(train, batch_size, augmentation, dataloader_kwargs):
    if list(augmentation) == []:
        transform = transforms.Compose([
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
            lambda x: 1-x,
        ])
    elif list(augmentation) == ['crop']:
        transform = transforms.Compose([
            transforms.Resize(size=(28, 28)),
            transforms.RandomCrop(28, augmentation['crop']),
            transforms.ToTensor(),
            lambda x: 1-x,
        ])
    else:
        raise KeyError(f'Only crop augmentation supported. Got {augmentation}.')
        
    dset = datasets.omniglot.Omniglot(paths_config.location_dict['Omniglot'], download=True, background=train, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

def getloader_NotMNIST(train, batch_size, augmentation, dataloader_kwargs):
    if list(augmentation) == []:
        transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x[0].view(1, 28, 28) ,
        ])
    elif list(augmentation) == ['crop']:
        if augmentation['crop'] == 0:
            transform = transforms.Compose([
                transforms.ToTensor(),
                lambda x: x[0].view(1, 28, 28) ,
            ])
        else:
            raise ValueError(f'Crop not supported for NotMNIST')
    else:
        raise KeyError(f'Augmentation not supported. Got {augmentation}.')
        
    if train:
        raise ValueError(f'ImageNet- only existes for the validation data of ImgeNet exists. train is set to {train}, which is not allowed.')
        
    dset = datasets.ImageFolder('../datasets/notMNIST_small', transform=transform) # deleted corrupted files ['A'] = {'RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png'}, ['F'] = {'Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png'}
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader
    

def getloader_TINY(train, batch_size, augmentation, dataloader_kwargs, exclude_cifar=['H'], shuffle=None):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
                []
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )
        
    tiny_dset = TinyImages(transform=transform, exclude_cifar=exclude_cifar)
    if train:
        dset = torch.utils.data.Subset(tiny_dset, range(100000, 50000000))
        dset.__repr__ = tiny_dset.__repr__
    else:
        np.random.seed(seed=0)
        dset = torch.utils.data.Subset(tiny_dset, np.random.choice(100000, size=1000, replace=False))
        dset.__repr__ = tiny_dset.__repr__
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle if shuffle is not None else train,
        **dataloader_kwargs)
    return loader
    
def getloader_CIFAR10(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
                []
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )

    dset = datasets.CIFAR10(paths_config.location_dict['CIFAR10'], train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader
    
def getloader_CIFAR100(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
                []
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )

    dset = datasets.CIFAR100(paths_config.location_dict['CIFAR100'], train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader


def getloader_DownsampledImageNet(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
                []
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )

    folder = paths_config.location_dict['DownsampledImageNet']
    folder = folder + 'train/' if train else folder + 'val/'
    dset = datasets.ImageFolder(folder, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader


def getloader_SVHN(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        if augmentation.get('HFlip'):
            warnings.warn(f'Random horizontal flip augmentation selected for SVHN, which usually is not done. Augementations: {augmentation}')
        transform = transforms.Compose(
                []
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )
    if train:
        split = 'train'
    else:
        split = 'test'
    dset = datasets.SVHN(paths_config.location_dict['SVHN'], split=split, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

def getloader_Noise(train, batch_size, augmentation, dataloader_kwargs, noise_fn):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment', '224'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
                []
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )
    if augmentation.get('28g'):
        ground_ds = datasets.MNIST(paths_config.location_dict['MNIST'], train=train, download=True, transform=transform)
    elif augmentation.get('224'):
        ground_ds = imgnet.get_restrictedImageNet(train=train).dataset
    else:
        ground_ds = datasets.CIFAR10(paths_config.location_dict['CIFAR10'], train=train, download=True, transform=transform)
    
    dset = Noise_Dataset(ground_ds, noise_fn, label_fn=lambda x: 0, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader

def getloader_Uniform(train, batch_size, augmentation, dataloader_kwargs):
    return getloader_Noise(train, batch_size, augmentation, dataloader_kwargs, noisefunctions.noise_uniform)

def getloader_Smooth(train, batch_size, augmentation, dataloader_kwargs):
    return getloader_Noise(train, batch_size, augmentation, dataloader_kwargs, noisefunctions.noise_low_freq)

def getloader_Black(train, batch_size, augmentation, dataloader_kwargs):
    return getloader_Noise(train, batch_size, augmentation, dataloader_kwargs, noisefunctions.monochrome(0))

def getloader_LSUN_CR(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.Resize(size=(28, 28)),
                transforms.Grayscale(),
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
                [transforms.Resize(size=(32, 32)),]
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )
    if train:
        raise ValueError(f'Only the validation split of LSUN Classroom is available. train is set to {train}, which is not allowed.')
    else:
        classes = ['classroom_val']
    dset = datasets.LSUN(paths_config.location_dict['LSUN'], classes=classes, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader


def getloader_ImageNetM(train, batch_size, augmentation, dataloader_kwargs):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
                []
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )
    if train:
        raise ValueError(f'ImageNet- only existes for the validation data of ImgeNet exists. train is set to {train}, which is not allowed.')
    dset = datasets.ImageFolder(paths_config.location_dict['ImageNet-'], transform=transform)
    dset.__repr__ = lambda: "Dataset ImageNet-"
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader
    
    
def getloader_RImageNet(train, batch_size, augmentation, dataloader_kwargs):
    if len(augmentation)>0 and 'autoaugment' in augmentation and augmentation['autoaugment']:
        augm_type = 'autoaugment'
    elif len(augmentation)>0:
        augm_type = 'default'
    else:
        augm_type = 'none'
    return imgnet.get_restrictedImageNet(train=train, batch_size=batch_size, 
                                         shuffle=train, augm_type=augm_type,
                                         balanced=False, num_workers=dataloader_kwargs['num_workers'])
    
    
def getloader_NotRImageNet(train, batch_size, augmentation, dataloader_kwargs):
    if len(augmentation)>0 and 'autoaugment' in augmentation and augmentation['autoaugment']:
        augm_type = 'autoaugment'
    elif len(augmentation)>0:
        augm_type = 'default'
    else:
        augm_type = 'none'
    if 'sampler' in dataloader_kwargs:
        sampler = dataloader_kwargs['sampler']
    else:
        sampler = None
    return imgnet.get_restrictedImageNetOD(train=train, batch_size=batch_size, 
                                           shuffle=train, augm_type=augm_type,
                                           num_workers=dataloader_kwargs['num_workers'], sampler=sampler)


def getloader_Flowers(train, batch_size, augmentation, dataloader_kwargs):
    if len(augmentation)>0 and 'autoaugment' in augmentation and augmentation['autoaugment']:
        augm_type = 'autoaugment'
    else:
        augm_type = 'none'
    split = 'train' if train else 'test'
    if 'sampler' in dataloader_kwargs:
        sampler = dataloader_kwargs['sampler']
    else:
        sampler = None
    return flowers.get_flowers(split=split, batch_size=batch_size, shuffle=train, augm_type=augm_type,
                               num_workers=dataloader_kwargs['num_workers'], sampler=sampler)


def getloader_FGVC(train, batch_size, augmentation, dataloader_kwargs):
    if len(augmentation)>0 and 'autoaugment' in augmentation and augmentation['autoaugment']:
        augm_type = 'autoaugment'
    else:
        augm_type = 'none'
    split = 'train' if train else 'test'
    if 'sampler' in dataloader_kwargs:
        sampler = dataloader_kwargs['sampler']
    else:
        sampler = None
    return fgvc.get_fgvc_aircraft(split=split, batch_size=batch_size, shuffle=train, augm_type=augm_type,
                                  num_workers=dataloader_kwargs['num_workers'], sampler=sampler)

def getloader_cars(train, batch_size, augmentation, dataloader_kwargs):
    if len(augmentation)>0 and 'autoaugment' in augmentation and augmentation['autoaugment']:
        augm_type = 'autoaugment'
    else:
        augm_type = 'none'
    if 'sampler' in dataloader_kwargs:
        sampler = dataloader_kwargs['sampler']
    else:
        sampler = None
    return cars.get_stanford_cars(train=train, batch_size=batch_size, shuffle=train, augm_type=augm_type,
                                  num_workers=dataloader_kwargs['num_workers'], sampler=sampler)




class CustomDataSet(torch.utils.data.dataset.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        try:
            image = Image.open(img_loc).convert("RGB")
        except:
            print('Error loading: ' + img_loc)
            random.seed(idx)
            new_idx = random.randint(0, len(self.total_imgs))
            return self.__getitem__(new_idx)
        tensor_image = self.transform(image)
        return tensor_image, 0
    
    
def getloader_tiny_open_images(train, batch_size, augmentation, dataloader_kwargs, exclude_cifar=['H'], shuffle=None):
    if augmentation.get('28g'): #MNIST style format
        if not augmentation.keys() <= {'28g', 'crop'}:
            raise ValueError(f'Key list {list(augmentation)} of augmentation dict not supported. For 28g, only cropping can be used.')
        transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop(28, augmentation.get('crop', 0)),
                transforms.ToTensor(),
            ])
    else: #stay in 32x32 rgb format
        if not augmentation.keys() <= {'28g', 'crop', 'HFlip', 'autoaugment'}:
            raise ValueError(f'At least one of the augmentations {augmentation.keys()} is not supported.')
        transform = transforms.Compose(
                [transforms.Resize(size=(32, 32))]
                + augmentation.get('HFlip', False)*[transforms.RandomHorizontalFlip(),]
                + bool(augmentation.get('crop', 0))*[transforms.RandomCrop(32, augmentation.get('crop', 0)),]
                + augmentation.get('autoaugment', False)*[AutoAugment(),]
                + [transforms.ToTensor(),]
            )
    if train:
        split = 'train'
    else:
        split = 'test'
        
    folder = paths_config.location_dict['TinyOpen'] + split
    dset = CustomDataSet(folder, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train,
        **dataloader_kwargs)
    return loader
    
    
datasets_dict = {'MNIST':             getloader_MNIST,
                 'FashionMNIST':      getloader_FashionMNIST,
                 'EMNIST_Letters':    getloader_EMNIST_Letters,
                 'Omniglot':          getloader_Omniglot,
                 'NotMNIST':          getloader_NotMNIST,
                 'TINY':              getloader_TINY,
                 'CIFAR10':           getloader_CIFAR10,
                 'CIFAR100':          getloader_CIFAR100,
                 'SVHN':              getloader_SVHN,
                 'Uniform':           getloader_Uniform,
                 'RImgNet':           getloader_RImageNet,
                 'NotRImgNet':        getloader_NotRImageNet,
                 'Flowers':           getloader_Flowers,
                 'FGVC':              getloader_FGVC,
                 'Cars':              getloader_cars,
                 'Smooth':            getloader_Smooth,
                 'Black':             getloader_Black,
                 'LSUN_CR':           getloader_LSUN_CR,
                 'ImageNet-':         getloader_ImageNetM,
                 'TinyOpen':          getloader_tiny_open_images,
                 }


val_loader_out_dicts = dict([])


val_loader_out_dicts['CIFAR10'] = {
        'CIFAR100': dict([]),
        'SVHN': dict([]),
        'LSUN_CR': dict([]),
        'Uniform':  dict([]),
        #'ImageNet-':  dict([]),
        'Smooth':  dict([]),
        'TINY':  dict([]),
#         'TinyOpen': dict([])
        #'CIFAR10': dict([]),
}


val_loader_out_dicts['CIFAR100'] = {
        'CIFAR10': dict([]),
        'SVHN': dict([]),
        'LSUN_CR': dict([]),
        'Uniform':  dict([]),
#         'ImageNet-':  dict([]),
        'Smooth':  dict([]),
         'TINY':  dict([]),
#          'TinyOpen': dict([])
}


val_loader_out_dicts['RImgNet'] = {
        'Flowers': dict([]),
        'FGVC': dict([]),
        'Cars': dict([]),
        'Uniform': {'224': True},
        'Smooth': {'224': True},  
        'NotRImgNet': {'224': True}, 
}


def get_val_out_loaders(dset_in_name, batch_size, dataloader_kwargs):
    return [datasets_dict[name](train=False, batch_size=batch_size, augmentation=augm, 
                                dataloader_kwargs=dataloader_kwargs) for name, augm in val_loader_out_dicts[dset_in_name].items()]


def get_test_out_loaders(dset_in_name, batch_size, dataloader_kwargs):
    return {name : datasets_dict[name](train=False, batch_size=batch_size, augmentation=augm, 
                                       dataloader_kwargs=dataloader_kwargs) for name, augm in test_loader_out_dicts[dset_in_name].items()}
