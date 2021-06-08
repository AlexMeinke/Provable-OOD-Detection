import torch
import torch.nn as nn
import numpy as np


class NoneAugm(nn.Module):
    def forward(self, x):
        return x
    
    
class CutMix(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta
        
    def forward(self, x):

        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(x.size()[0]).to(x.device)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        return x
        
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2