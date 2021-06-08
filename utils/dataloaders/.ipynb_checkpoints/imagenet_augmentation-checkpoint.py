from torchvision import transforms
import torch
from utils.dataloaders.auto_augment import ImageNetPolicy

ImageNet_mean_int = ( int( 255 * 0.485), int(255 * 0.456), int(255 * 0.406))


def get_imageNet_augmentation(type='default', out_size=224, config_dict=None):
    if type == 'none' or type is None:
        transform_list = [
            transforms.Resize((out_size,out_size)),
            transforms.ToTensor()
        ]
        transform = transforms.Compose(transform_list)
        return transform
    elif type == 'default':
        transform_list = [
            transforms.transforms.RandomResizedCrop(out_size),
            transforms.RandomHorizontalFlip(),
        ]
    elif type == 'autoaugment':
        transform_list = [
            transforms.RandomResizedCrop(out_size),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(fillcolor=ImageNet_mean_int),
        ]
    else:
        raise ValueError(f'augmentation type - {type} - not supported')

    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    if config_dict is not None:
        config_dict['type'] = type
        config_dict['Output out_size'] = out_size

    return transform