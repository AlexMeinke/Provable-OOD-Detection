import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Add_ParamI(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = x + self.bias
        return out
        
    def ibp_forward(self, l, u):
        l_ = l + self.bias
        u_ = u + self.bias
        return l_, u_
                
class Scale_By_ParamI(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = x * self.scalar
        return out
    
    def ibp_forward(self, l, u):
        if self.scalar >= 0:
            l_ = l * self.scalar
            u_ = u * self.scalar
        else:
            u_ = l * self.scalar
            l_ = u * self.scalar
        return l_, u_

class FlattenI(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    def ibp_forward(self, l, u):
        l_ = self.forward(l)
        u_ = self.forward(u)
        return l_, u_
    
class Conv2dI(nn.Conv2d):    
    def ibp_forward(self, l, u):
        #print(type(child))
        l_ = (nn.functional.conv2d(l, self.weight.clamp(min=0), bias=None, 
                                   stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, groups=self.groups) +
              nn.functional.conv2d(u, self.weight.clamp(max=0), bias=None, 
                                   stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, groups=self.groups)
             )
        u_ = (nn.functional.conv2d(u, self.weight.clamp(min=0), bias=None, 
                                   stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, groups=self.groups) +
              nn.functional.conv2d(l, self.weight.clamp(max=0), bias=None, 
                                   stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, groups=self.groups)
             )
        if self.bias is not None:
            l_ += self.bias[None,:,None,None]
            u_ += self.bias[None,:,None,None]
        return l_, u_

class ReLUI(nn.ReLU):
    def ibp_forward(self, l, u):
        l_ = l.clamp(min=0)
        u_ = u.clamp(min=0)
        return l_, u_
    
class AvgPool2dI(nn.AvgPool2d):
    def ibp_forward(self, l, u):
        return self.forward(l), self.forward(u)
    
class AdaptiveAvgPool2dI(nn.AdaptiveAvgPool2d):
    def ibp_forward(self, l, u):
        return self.forward(l), self.forward(u)
    
class MaxPool2dI(nn.MaxPool2d):
    def ibp_forward(self, l, u):
        return self.forward(l), self.forward(u)
    
class DropoutI(nn.Dropout):
    def ibp_forward(self, l, u):
        assert not self.training, "IBP pass only implemented in eval mode!"
        return l, u
    
class BatchNorm2dI(nn.BatchNorm2d):
    def ibp_forward(self, l, u):
        assert not self.training, "IBP pass only implemented in eval mode!"
        assert self.track_running_stats, "track_running_stats must be set to true for the IBP forward pass!"
        idx = (self.weight>=0).float()[None,:,None,None]
        lp = l*idx + u*(1-idx)
        up = u*idx + l*(1-idx)
        return self.forward(lp), self.forward(up)


class LinearI(nn.Linear):
    def ibp_forward(self, l, u):
        if self.bias is not None:
            l_ = (self.weight.clamp(min=0) @ l.t() + self.weight.clamp(max=0) @ u.t() + self.bias[:,None]).t()
            u_ = (self.weight.clamp(min=0) @ u.t() + self.weight.clamp(max=0) @ l.t() + self.bias[:,None]).t()
        else:
            l_ = (self.weight.clamp(min=0) @ l.t() + self.weight.clamp(max=0) @ u.t()).t()
            u_ = (self.weight.clamp(min=0) @ u.t() + self.weight.clamp(max=0) @ l.t()).t()
        return l_, u_
    
    
class LinearI_Neg(nn.Linear):
    def forward(self, x):
        return F.linear(x, -self.weight.exp(), self.bias)
        
    def ibp_forward(self, l, u):
        weight = -self.weight.exp()
        if self.bias is not None:
            l_ = (weight.clamp(min=0) @ l.t() + weight.clamp(max=0) @ u.t() + self.bias[:,None]).t()
            u_ = (weight.clamp(min=0) @ u.t() + weight.clamp(max=0) @ l.t() + self.bias[:,None]).t()
        else:
            l_ = (weight.clamp(min=0) @ l.t() + weight.clamp(max=0) @ u.t()).t()
            u_ = (weight.clamp(min=0) @ u.t() + weight.clamp(max=0) @ l.t()).t()
        return l_, u_
    
    
def conv3x3I(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2dI(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    
    
class SequentialI(nn.Sequential):
    def ibp_forward(self, l, u):
        for module in self:
            l,u = module.ibp_forward(l, u)
        return l,u

    
def elision(l, u, layer, num_classes=10):
    assert isinstance(layer, LinearI)
    W = layer.weight

    Wd = W.unsqueeze(dim=1).expand((num_classes, num_classes, -1)) - W.unsqueeze(dim=0).expand(
        (num_classes, num_classes, -1))

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

    l, u = l_, u_
    
    return l, u, ud
    
    
class JointModel(nn.Module):
    def __init__(self, base_model, detector, device=torch.device('cpu'), classes=None):
        super().__init__()
        assert classes is not None
        self.base_model = base_model
        self.detector = detector
        self.classes = classes
        self.log_classes = torch.log(torch.tensor(self.classes, dtype=torch.float)).item()
        
    def forward(self, x):
        zero = torch.tensor(0., device=x.device)
        log_pred = torch.log_softmax(self.base_model(x), dim=1)
        det = self.detector(x)
        log_pi = - torch.logaddexp(zero, -det)
        log_po = - torch.logaddexp(zero, det)
        result = torch.logaddexp(log_pi + log_pred, log_po - self.log_classes)
        return result
    
    def ibp_forward(self, l, u):
        l, u = self.detector.ibp_forward(l, u)
        p_in = 1./(1 + (-u).exp()).squeeze()
        return None, ( p_in + (1.-p_in) / self.classes ).log()
    