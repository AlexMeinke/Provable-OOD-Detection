import torch
import torch.nn as nn
import copy
from torch.nn import functional as F


import utils.models.modules_ibp as modules_ibp


class CNN_IBP(nn.Module):
    def __init__(self, dset_in_name='MNIST', size='L', width=None, last_bias=True, num_classes=None, last_layer_neg=False):
        super().__init__()
        if dset_in_name == 'MNIST':
            self.color_channels = 1
            self.hw = 28
            num_classes = 10 if num_classes is None else num_classes
        elif dset_in_name == 'CIFAR10' or dset_in_name == 'SVHN':
            self.color_channels = 3
            self.hw = 32
            num_classes = 10 if num_classes is None else num_classes
        elif dset_in_name == 'CIFAR100':
            self.color_channels = 3
            self.hw = 32
            num_classes = 100 if num_classes is None else num_classes
        elif dset_in_name == 'RImgNet':
            self.color_channels = 3
            self.hw = 224
            num_classes = 9 if num_classes is None else num_classes
        else:
            raise ValueError(f'{dset_in_name} dataset not supported.')
        self.num_classes = num_classes
        self.size = size
        self.width = width
        
        if last_layer_neg:
            last_layer_type = modules_ibp.LinearI_Neg
        else:
            last_layer_type = modules_ibp.LinearI
        self.last_layer_type = last_layer_type
            
        if size == 'L':   
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 64, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(64, 64, 3, padding=1, stride=1)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(64, 128, 3, padding=1, stride=2)
            self.A3 = modules_ibp.ReLUI()
            self.C4 = modules_ibp.Conv2dI(128, 128, 3, padding=1, stride=1)
            self.A4 = modules_ibp.ReLUI()
            self.C5 = modules_ibp.Conv2dI(128, 128, 3, padding=1, stride=1)
            self.A5 = modules_ibp.ReLUI()
            self.F = modules_ibp.FlattenI()
            self.L6 = modules_ibp.LinearI(128*(self.hw//2)**2, 512)
            self.A6 = modules_ibp.ReLUI()
            self.L7 = last_layer_type(512, self.num_classes, bias=last_bias)

            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.C4,
                           self.A4,
                           self.C5,
                           self.A5,
                           self.F,
                           self.L6,
                           self.A6,
                           self.L7,
                          )

            self.__name__ = 'CNN_L_' + dset_in_name

        elif size == 'XL':   
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 128, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(128, 128, 3, padding=1, stride=1)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(128, 256, 3, padding=1, stride=2)
            self.A3 = modules_ibp.ReLUI()
            self.C4 = modules_ibp.Conv2dI(256, 256, 3, padding=1, stride=1)
            self.A4 = modules_ibp.ReLUI()
            self.C5 = modules_ibp.Conv2dI(256, 256, 3, padding=1, stride=1)
            self.A5 = modules_ibp.ReLUI()
            self.F = modules_ibp.FlattenI()
            self.L6 = modules_ibp.LinearI(256*(self.hw//2)**2, 512)
            self.A6 = modules_ibp.ReLUI()
            self.L7 = modules_ibp.LinearI(512, 512)
            self.A7 = modules_ibp.ReLUI()
            self.L8 = last_layer_type(512, self.num_classes, bias=last_bias)
                        
            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.C4,
                           self.A4,
                           self.C5,
                           self.A5,
                           self.F,
                           self.L6,
                           self.A6,
                           self.L7,
                           self.A7,
                           self.L8,
                          )

            self.__name__ = 'CNN_XL_' + dset_in_name
            
        elif size == 'XL_b':   
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 128, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(128, 128, 3, padding=1, stride=1)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(128, 256, 3, padding=1, stride=2)
            self.A3 = modules_ibp.ReLUI()
            self.C4 = modules_ibp.Conv2dI(256, 256, 3, padding=1, stride=1)
            self.A4 = modules_ibp.ReLUI()
            self.C5 = modules_ibp.Conv2dI(256, 256, 3, padding=1, stride=1)
            self.A5 = modules_ibp.ReLUI()
            self.F = modules_ibp.FlattenI()
            self.L6 = modules_ibp.LinearI(256*(self.hw//2)**2, 512)
            self.A6 = modules_ibp.ReLUI()
            self.L7 = last_layer_type(512, self.num_classes, bias=last_bias)
                        
            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.C4,
                           self.A4,
                           self.C5,
                           self.A5,
                           self.F,
                           self.L6,
                           self.A6,
                           self.L7,
                          )

            self.__name__ = 'CNN_XL_b_' + dset_in_name

        elif size == 'C1':   
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 128, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(128, 128, 3, padding=1, stride=1)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(128, 256, 3, padding=1, stride=2)
            self.A3 = modules_ibp.ReLUI()
            self.C4 = modules_ibp.Conv2dI(256, 256, 3, padding=1, stride=1)
            self.A4 = modules_ibp.ReLUI()
            self.C5 = modules_ibp.Conv2dI(256, 256, 3, padding=1, stride=1)
            self.A5 = modules_ibp.ReLUI()
            self.F = modules_ibp.FlattenI()
            self.L6 = modules_ibp.LinearI(256*(self.hw//2)**2, 512)
            self.A6 = modules_ibp.ReLUI()
            self.L7 = last_layer_type(512, self.num_classes)
                        
            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.C4,
                           self.A4,
                           self.C5,
                           self.A5,
                           self.F,
                           self.L6,
                           self.A6,
                           self.L7,
                          )

            self.__name__ = 'CNN_C1_' + dset_in_name
        
        elif size == 'C2':   
            self.width = 2
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 128*self.width, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(128*self.width, 128*self.width, 3, padding=1, stride=1)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(128*self.width, 256*self.width, 3, padding=1, stride=2)
            self.A3 = modules_ibp.ReLUI()
            self.C4 = modules_ibp.Conv2dI(256*self.width, 256*self.width, 3, padding=1, stride=1)
            self.A4 = modules_ibp.ReLUI()
            self.C5 = modules_ibp.Conv2dI(256*self.width, 256*self.width, 3, padding=1, stride=1)
            self.A5 = modules_ibp.ReLUI()
            self.F = modules_ibp.FlattenI()
            self.L6 = modules_ibp.LinearI(256*self.width*(self.hw//2)**2, 512*self.width)
            self.A6 = modules_ibp.ReLUI()
            self.L7 = last_layer_type(512*self.width, self.num_classes)
                        
            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.C4,
                           self.A4,
                           self.C5,
                           self.A5,
                           self.F,
                           self.L6,
                           self.A6,
                           self.L7,
                          )

            self.__name__ = f'CNN_C2-{self.width}_' + dset_in_name

        
        elif size == 'C3':
            self.width = 2
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 128*self.width, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(128*self.width, 256*self.width, 3, padding=1, stride=2)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(256*self.width, 256*self.width, 3, padding=1, stride=1)
            self.A3 = modules_ibp.ReLUI()
            self.F = modules_ibp.FlattenI()
            self.L4 = modules_ibp.LinearI(256*self.width*(self.hw//2)**2, 512)
            self.A4 = modules_ibp.ReLUI()
            self.L5 = last_layer_type(512, self.num_classes)
                        
            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.F,
                           self.L4,
                           self.A4,
                           self.L5,
                          )

            self.__name__ = f'CNN_C3-{self.width}_' + dset_in_name
            
        elif size == 'C3s':
            self.width = 2
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 128*self.width, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(128*self.width, 256*self.width, 3, padding=1, stride=2)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(256*self.width, 256*self.width, 3, padding=1, stride=1)
            self.A3 = modules_ibp.ReLUI()
            self.pool = modules_ibp.AvgPool2dI(2)
            self.F = modules_ibp.FlattenI()
            self.L4 = modules_ibp.LinearI(256*self.width*(self.hw//4)**2, 128)
            self.A4 = modules_ibp.ReLUI()
            self.L5 = last_layer_type(128, self.num_classes)
                        
            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.pool,
                           self.F,
                           self.L4,
                           self.A4,
                           self.L5,
                          )

            self.__name__ = f'CNN_C3s-{self.width}_' + dset_in_name
            
        elif size == 'S':
            self.width = 1
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 128*self.width, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(128*self.width, 256*self.width, 3, padding=1, stride=2)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(256*self.width, 256*self.width, 3, padding=1, stride=1)
            self.A3 = modules_ibp.ReLUI()
            self.pool = modules_ibp.AvgPool2dI(2)
            self.F = modules_ibp.FlattenI()
            self.L4 = modules_ibp.LinearI(256*self.width*(self.hw//4)**2, 128)
            self.A4 = modules_ibp.ReLUI()
            self.L5 = last_layer_type(128, self.num_classes)
                        
            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.pool,
                           self.F,
                           self.L4,
                           self.A4,
                           self.L5,
                          )

            self.__name__ = f'CNN_S-{self.width}_' + dset_in_name
            
        elif size == 'SR':
            self.width = 1
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 128*self.width, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.pool1 = modules_ibp.AvgPool2dI(2)
            self.C2 = modules_ibp.Conv2dI(128*self.width, 256*self.width, 3, padding=1, stride=2)
            self.A2 = modules_ibp.ReLUI()
            self.pool2 = modules_ibp.AvgPool2dI(2)
            self.C3 = modules_ibp.Conv2dI(256*self.width, 256*self.width, 3, padding=1, stride=1)
            self.A3 = modules_ibp.ReLUI()
            self.pool3 = modules_ibp.AvgPool2dI(2)
            self.F = modules_ibp.FlattenI()
            self.L4 = modules_ibp.LinearI(256*self.width*(self.hw//16)**2, 128)
            self.A4 = modules_ibp.ReLUI()
            self.L5 = last_layer_type(128, self.num_classes)
                        
            self.layers = nn.ModuleList([self.C1,
                           self.A1,
                           self.pool1,
                           self.C2,
                           self.A2,
                           self.pool2,
                           self.C3,
                           self.A3,
                           self.pool3,
                           self.F,
                           self.L4,
                           self.A4,
                           self.L5,
                                        ])

            self.__name__ = f'CNN_S-{self.width}_' + dset_in_name
            
        elif size == 'SR2':
            self.width = 1
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 128*self.width, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(128*self.width, 256*self.width, 3, padding=1, stride=2)
            self.A2 = modules_ibp.ReLUI()
            self.pool2 = modules_ibp.AvgPool2dI(2)
            self.C3 = modules_ibp.Conv2dI(256*self.width, 256*self.width, 3, padding=1, stride=1)
            self.A3 = modules_ibp.ReLUI()
            self.pool3 = modules_ibp.AvgPool2dI(2)
            self.F = modules_ibp.FlattenI()
            self.L4 = modules_ibp.LinearI(256*self.width*(self.hw//8)**2, 128)
            self.A4 = modules_ibp.ReLUI()
            self.L5 = last_layer_type(128, self.num_classes)
                        
            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.pool2,
                           self.C3,
                           self.A3,
                           self.pool3,
                           self.F,
                           self.L4,
                           self.A4,
                           self.L5,
                          )

            self.__name__ = f'CNN_S-{self.width}_' + dset_in_name
            
        elif size == 'XS':
            self.width = 1
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 64*self.width, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(64*self.width, 128*self.width, 3, padding=1, stride=2)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(128*self.width, 128*self.width, 3, padding=1, stride=1)
            self.A3 = modules_ibp.ReLUI()
            self.pool = modules_ibp.AvgPool2dI(2)
            self.F = modules_ibp.FlattenI()
            self.L4 = modules_ibp.LinearI(128*self.width*(self.hw//4)**2, 128)
            self.A4 = modules_ibp.ReLUI()
            self.L5 = last_layer_type(128, self.num_classes)
                        
            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.pool,
                           self.F,
                           self.L4,
                           self.A4,
                           self.L5,
                          )

            self.__name__ = f'CNN_S-{self.width}_' + dset_in_name
        
        elif size == 'C4':
            self.width = 1
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 512*self.width, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(512*self.width, 1024*self.width, 3, padding=1, stride=2)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(1024*self.width, 512*self.width, 3, padding=1, stride=1)
            self.A3 = modules_ibp.ReLUI()
            self.F = modules_ibp.FlattenI()
            self.L4 = modules_ibp.LinearI(512*self.width*(self.hw//2)**2, 128)
            self.A4 = modules_ibp.ReLUI()
            self.L5 = last_layer_type(128, self.num_classes)
                        
            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.F,
                           self.L4,
                           self.A4,
                           self.L5,
                          )

            self.__name__ = f'CNN_C4-{self.width}_' + dset_in_name
        
        elif size == 'C5':   
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 512, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(512, 512, 3, padding=1, stride=1)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(512, 512, 3, padding=1, stride=2)
            self.A3 = modules_ibp.ReLUI()
            self.C4 = modules_ibp.Conv2dI(512, 512, 3, padding=1, stride=1)
            self.A4 = modules_ibp.ReLUI()
            self.C5 = modules_ibp.Conv2dI(512, 256, 3, padding=1, stride=1)
            self.A5 = modules_ibp.ReLUI()
            self.F = modules_ibp.FlattenI()
            self.L6 = modules_ibp.LinearI(256*(self.hw//2)**2, 512)
            self.A6 = modules_ibp.ReLUI()
            self.L7 = last_layer_type(512, self.num_classes)
                        
            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.C4,
                           self.A4,
                           self.C5,
                           self.A5,
                           self.F,
                           self.L6,
                           self.A6,
                           self.L7,
                          )

            self.__name__ = 'CNN_C5_' + dset_in_name
            
        elif size == 'C6':   
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 128, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(128, 128, 3, padding=1, stride=1)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(128, 256, 3, padding=1, stride=2)
            self.A3 = modules_ibp.ReLUI()
            self.C4 = modules_ibp.Conv2dI(256, 256, 3, padding=1, stride=1)
            self.A4 = modules_ibp.ReLUI()
            self.C5 = modules_ibp.Conv2dI(256, 128, 3, padding=1, stride=1)
            self.A5 = modules_ibp.ReLUI()
            self.F = modules_ibp.FlattenI()
            self.L6 = modules_ibp.LinearI(128*(self.hw//2)**2, 256)
            self.A6 = modules_ibp.ReLUI()
            self.L7 = last_layer_type(256, self.num_classes)
                        
            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.C4,
                           self.A4,
                           self.C5,
                           self.A5,
                           self.F,
                           self.L6,
                           self.A6,
                           self.L7,
                          )
            
            self.__name__ = 'CNN_C6_' + dset_in_name
            
        elif size == 'C7':   
            self.C1 = modules_ibp.Conv2dI(self.color_channels, 256, 3, padding=1, stride=1)
            self.A1 = modules_ibp.ReLUI()
            self.C2 = modules_ibp.Conv2dI(256, 256, 3, padding=1, stride=1)
            self.A2 = modules_ibp.ReLUI()
            self.C3 = modules_ibp.Conv2dI(256, 512, 3, padding=1, stride=2)
            self.A3 = modules_ibp.ReLUI()
            self.C4 = modules_ibp.Conv2dI(512, 512, 3, padding=1, stride=1)
            self.A4 = modules_ibp.ReLUI()
            self.C5 = modules_ibp.Conv2dI(512, 128, 3, padding=1, stride=1)
            self.A5 = modules_ibp.ReLUI()
            self.F = modules_ibp.FlattenI()
            self.L6 = modules_ibp.LinearI(128*(self.hw//2)**2, 256)
            self.A6 = modules_ibp.ReLUI()
            self.L7 = last_layer_type(256, self.num_classes)
                        
            self.layers = (self.C1,
                           self.A1,
                           self.C2,
                           self.A2,
                           self.C3,
                           self.A3,
                           self.C4,
                           self.A4,
                           self.C5,
                           self.A5,
                           self.F,
                           self.L6,
                           self.A6,
                           self.L7,
                          )
            
            self.__name__ = 'CNN_C6_' + dset_in_name
        
        else:
            raise ValueError(str(size) + ' is not a valid size.')
        
    def forward(self, x):
        x = x.type(torch.get_default_dtype())
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def forward_layer(self, x, idx):
        x = x.type(torch.get_default_dtype())
        for layer in self.layers[:idx]:
            x = layer.forward(x)
        return x
    
    def forward_layer_list(self, x, layers):
        bs = x.shape[0]
        output = []
        x = x.type(torch.get_default_dtype())
        for idx, layer in enumerate(self.layers[:max(layers)+1]):
            if idx in layers:
                output.append(x.view(bs, -1))
                
            x = layer.forward(x)
            
        return torch.cat(output, 1)
    
    def ibp_forward(self, l, u):
        l = l.type(torch.get_default_dtype())
        u = u.type(torch.get_default_dtype())
        for layer in self.layers:
            l, u = layer.ibp_forward(l, u)
        return l, u
    
    def ibp_forward_layer(self, l, u, idx):
        l = l.type(torch.get_default_dtype())
        u = u.type(torch.get_default_dtype())
        for layer in self.layers[:idx]:
            l, u = layer.ibp_forward(l, u)
        return l, u
    
    def ibp_elision_forward(self, l, u, num_classes):
        l = l.type(torch.get_default_dtype())
        u = u.type(torch.get_default_dtype())
        for layer in self.layers[:-1]:
            l, u = layer.ibp_forward(l, u)
        
        layer = self.layers[-1]
        assert isinstance(layer, modules_ibp.LinearI)
        W = layer.weight
        Wd = W.unsqueeze(dim=1).expand((num_classes,num_classes,-1)) - W.unsqueeze(dim=0).expand((num_classes,num_classes,-1))
        ud = torch.einsum('abc,nc->nab', Wd.clamp(min=0), u) + torch.einsum('abc,nc->nab', Wd.clamp(max=0), l)
        if layer.bias is not None:
            bd = layer.bias.unsqueeze(dim=1).expand((num_classes,num_classes)) -  layer.bias.unsqueeze(dim=0).expand((num_classes,num_classes))
            ud += bd.unsqueeze(0)

        if layer.bias is not None:
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t() + layer.bias[:,None]).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t() + layer.bias[:,None]).t()
        else:
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t()).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t()).t()
        l,u = l_, u_
        return l, u, ud
    
    def make_table(self):
        sl = []
        for l in self.layers:
            pass
        
    def forward_pre_logit(self, x):
        x = x.type(torch.get_default_dtype())
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        return self.layers[-1](x), x.detach().cpu()
    
#     def rescale(self):
#         EPS = 1e-8
#         scale1 = self.layers[13].weight.data.abs().mean(1) + EPS
#         scale2 = self.layers[15].weight.data.abs().mean(0) + EPS
        
#         r = (scale2 / scale1 ) ** .5

#         w1 = self.layers[13].weight.data * r[:,None]
#         b1 = self.layers[13].bias.data * r
#         w2 = self.layers[15].weight.data / r[None,:]

#         self.layers[13].weight.data = w1
#         self.layers[15].weight.data = w2
#         self.layers[13].bias.data = b1
        
        
#         scale1 = self.layers[11].weight.data.abs().mean(1) + EPS
#         scale2 = self.layers[13].weight.data.abs().mean(0) + EPS

#         r = (scale2 / scale1 + 1e-8) ** .5

#         w1 = self.layers[11].weight.data * r[:,None]
#         b1 = self.layers[11].bias.data * r
#         w2 = self.layers[13].weight.data / r[None,:]

#         self.layers[11].weight.data = w1
#         self.layers[13].weight.data = w2
#         self.layers[11].bias.data = b1
        
    def rescale(self):
        s = []
        layer_idx = [0, 2, 4, 6, 8, 11, 13, 15]
        for idx in layer_idx[:-1]:
            a = self.layers[idx].weight.data.abs().mean()
            s.append(a.log().item())
        s = torch.tensor(s)

        s8 = self.layers[15].weight.data.abs().mean().log().item()
        fudge_factor = 1.
        beta = fudge_factor*(s.sum() - 7*s8) / 8

        for i, idx in enumerate(layer_idx[:-1]):
            self.layers[idx].weight.data *= (-beta/7).exp()
            self.layers[idx].bias.data *= (-(i+1)*beta/7).exp()
        self.layers[15].weight.data *= beta.exp()
        
