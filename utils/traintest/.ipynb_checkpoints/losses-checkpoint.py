import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GOODLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, ub_log_conf):
        return ((ub_log_conf)**2/2).log1p()
    
    
class BinaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, output):
        return torch.logaddexp(torch.tensor([1.0], device=output.device), -output)

    
class SavageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, output):
        return 1 / ( 1 + output.exp() )**2
    
    
class BinaryMarginLoss(nn.Module):
    def __init__(self, margin=.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, output):
        return torch.logaddexp(torch.tensor([1.0], device=output.device), self.margin-output)
    
    
class HingeLoss(nn.Module):
    def __init__(self, margin=1.):
        super().__init__()
        self.margin = margin
        
    def forward(self, output):
        return F.relu(self.margin-output)


class QGOODLoss(nn.Module):
    def __init__(self, quantile=0.8):
        super().__init__()
        self.quantile = quantile
        
    def forward(self, ub_log_conf):
        
        batch_size_out = ub_log_conf.shape[0]
        l = math.floor(batch_size_out*self.quantile)
        h = batch_size_out - l
    
        above_quantile_indices = ub_log_conf.topk(h, largest=True)[1] #above or exactly at quantile, i.e. 'not below'.
        # below_quantile_indices = ub_log_conf_out_batch.topk(l, largest=False)[1]
        return ((ub_log_conf[above_quantile_indices])**2/2).log1p()
    
    
class DirectLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, ub_log_conf):
        return ub_log_conf
    
    
class OELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, logits):
        return -torch.log_softmax(logits, 1).mean(1)
    
    
class OELossLogConf(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, confs):
        return -confs.mean(1)

    
loss_dict = {
    'GOOD': GOODLoss,
    'QGOOD': QGOODLoss,
    'direct': DirectLoss,
    'binary': BinaryLoss,
    'savage': SavageLoss,
    'binary_margin': BinaryMarginLoss,
    'margin': HingeLoss,
    'ce': nn.CrossEntropyLoss,
    'ce_logconf': nn.NLLLoss,
    'oe': OELoss,
    'oe_logconf': OELossLogConf,
}