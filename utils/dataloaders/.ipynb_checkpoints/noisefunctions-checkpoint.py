import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import PIL
import itertools
import datetime
import random
import skimage
from skimage import filters


def noise_permute(datapoint):
    """Permutes the pixels of an img and assigns the label (label, 'permuted').
        The input should be an image (PIL, others like numpy arrays might work, too) with a label.
        The returned image is a PIL image.
        It is assumed that img has 3 dimensions, the last of which is the color channels.
    """
    img, label = datapoint
    imgn = np.transpose(img.numpy(), (1,2,0))
    assert len(imgn.shape) == 3 and imgn.shape[2] <=4, 'Unexpected image dimensions.'
    imgn_flat = imgn.reshape(imgn.shape[0]*imgn.shape[1], imgn.shape[2])
    imgn_flat_permuted = np.random.permutation(imgn_flat) #this function shuffles the first axis
    imgn_permuted = imgn_flat_permuted.reshape(imgn.shape)
    return torch.from_numpy(np.transpose(imgn_permuted, (2,0,1))), label #(label, 'permuted')


def filter_gauss(datapoint, srange=[1,1]):
    img, label = datapoint
    imgn = np.transpose(img.numpy(), (1,2,0))
    sigma = srange[0] + np.random.random_sample()*(srange[1]-srange[0])
    imgn_gaussed = skimage.filters.gaussian(imgn, sigma=sigma, multichannel=3)
    return torch.from_numpy(np.transpose(imgn_gaussed, (2,0,1))), label #+ ('gauss', sigma)

def gaussed_noise_perm(x):
    x = noise_permute(x)
    x = filter_gauss(x, srange=[0.25,1.25])
    return x

def scale_full_range(datapoint):
    img_in = datapoint[0]
    img_0_based = img_in - img_in.min()
    img_scaled = img_0_based/(img_0_based.max())
    return img_scaled, datapoint[1]

def noise_uniform(datapoint):
    """Returns uniform noise with the same shape as the input.
        The input should be an image (PIL, others like numpy arrays might work, too) with a label.
        The returned image is a PIL image.
        It is assumed that img has 3 dimensions, the last of which is the color channels.
    """
    img, label = datapoint
    assert len(img.shape) == 3, 'Unexpected image dimensions:' + str(img.shape)
    imgn = np.transpose(img.numpy(), (1,2,0))
    if imgn.shape[2] != 1:
        assert imgn.shape[2] == 3, 'Unexpected last image dimensions:' + str(imgn.shape)
        imgn_random = np.float32(np.random.uniform(size=imgn.shape))
        return torch.from_numpy(np.transpose(imgn_random, (2,0,1))), label 
    else:
        imgn_random = np.float32(np.random.uniform(size=imgn.shape))
        assert torch.from_numpy(np.transpose(imgn_random, (2,0,1))).shape == img.shape, 'torch.from_numpy(np.transpose(imgn_random, (2,0,1))).shape wrong: ' + str(torch.from_numpy(np.transpose(imgn_random, (2,0,1))).shape)
        return torch.from_numpy(np.transpose(imgn_random, (2,0,1))), label 

def noise_low_freq(datapoint):
    uniform = noise_uniform(datapoint)
    gaussed = filter_gauss(uniform, srange=[1,2.5])
    low_freq = scale_full_range(gaussed)
    return low_freq

def identity(datapoint):
    return datapoint
    
class monochrome:
    def __init__(self, color):
        super().__init__()
        self.color = color
    
    def __call__(self, datapoint):
        img, label = datapoint
        assert len(img.shape) == 3, 'Unexpected image dimensions:' + str(img.shape)
        imgn = np.transpose(img.numpy(), (1,2,0))
        imgn_monochrome = np.float32(self.color*np.ones(imgn.shape))
        return torch.from_numpy(np.transpose(imgn_monochrome, (2,0,1))), label
    
class uniform_on_sphere:
    def __init__(self, radius, center):
        super().__init__()
        self.radius = radius
        self.center = center
    
    def draw(self, datapoint):
        img, label = datapoint
        normal_rand_img = torch.randn(img.shape)
        scaling = self.radius / normal_rand_img.norm(2)
        return normal_rand_img * scaling + self.center, label

class rectangles:
    def __init__(self, sections):
        super().__init__()
        self.h_sections, self.v_sections = sections
        
    def draw(self, datapoint):
        img, label = datapoint
        assert img.shape[1] % self.h_sections == 0, 'horizontal tiling invalid'
        assert img.shape[2] % self.v_sections == 0, 'vertical tiling invalid'
        h_section_length = img.shape[1] // self.h_sections
        v_section_length = img.shape[2] // self.v_sections
        
        h = random.randint(0, self.h_sections-1)
        v = random.randint(0, self.v_sections-1)
        img *= 0
        img[:,h*h_section_length:(h+1)*h_section_length,v*v_section_length:(v+1)*v_section_length] = 1
        return img, label

