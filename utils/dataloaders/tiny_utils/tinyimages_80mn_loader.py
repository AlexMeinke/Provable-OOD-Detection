import numpy as np
import torch
from bisect import bisect_left
from PIL import Image
import os

# from .tiny_path_config import tiny_path
import paths_config

tiny_path = paths_config.location_dict['TinyImages']

dirname = os.path.dirname(__file__)

class TinyImages(torch.utils.data.Dataset):

    def __init__(self, transform=None, exclude_cifar=['H','CEDA11']):

        self.data_file = open(tiny_path, "rb")

        def load_image(idx):
            self.data_file.seek(idx * 3072)
            data = self.data_file.read(3072)
            return Image.fromarray(np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F"))

        self.load_image = load_image
        self.offset = 0     # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar
        
        self.cifar_idxs = []
        
        if 'H' in exclude_cifar:
            with open(os.path.join(dirname, '80mn_cifar_idxs.txt'), 'r') as idxs:
                for idx in idxs:
                   # indices in file take the 80mn database to start at 1, hence "- 1"
                   self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            # bisection search option
            # self.cifar_idxs = tuple(sortedu -hs /path/to/directoryd(self.cifar_idxs))
            #
            # def binary_search(x, hi=len(self.cifar_idxs)):
            #     pos = bisect_left(self.cifar_idxs, x, 0, hi)  # find insertion position
            #     return True if pos != hi and self.cifar_idxs[pos] == x else False
            #
            # self.in_cifar = binary_search
        
        if 'CEDA11' in exclude_cifar:
            for i in range(80):
                with open(os.path.join('tiny_utils/exclude_cifar/', f'{i:02d}M'), 'r') as idxs:
                    for idx in idxs:
                        # indices in file take the 80mn database to start at 1, hence "- 1"
                        self.cifar_idxs.append(int(idx.split()[0]))
        
        self.cifar_idxs = set(self.cifar_idxs)
        self.in_cifar = lambda x: x in self.cifar_idxs
            
    def __getitem__(self, index):
        index = (index + self.offset) % 79302016

        while self.in_cifar(index):
            index = np.random.randint(79302017)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return 79302017
    
    _repr_indent = 4
    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()} - {len(self.cifar_idxs)}"]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def reopen_data_file(self):
        self.data_file.close()
        self.data_file = open(tiny_path, "rb")