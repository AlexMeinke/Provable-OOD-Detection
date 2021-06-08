import os
from tinydb import TinyDB
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

import utils.models.provable_classifiers as provable_classifiers
import utils.dataloaders.dataloading as dataloading
import utils.traintest.better_training as better
import utils.traintest.evaluation as ev
import utils.models.hendrycks as hendrycks
import utils.models.resnet as rn
import utils.models.modules_ibp as ibp
import paths_config
import utils.factories as fac


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    
# this will be the main way to load models
def from_database(doc_id, dataset, device=torch.device('cpu')):
    file = paths_config.project_folder + 'evals/' + dataset + '.json'
    try:
        db = TinyDB(file)
    except:
        db = TinyDB(os.getcwd() + '/' + file)
    data = {}
    
    entry = db.get(doc_id=doc_id)
    train_args = entry['args']
    
    model = fac.get_model(dotdict(train_args['architecture'])).to(device)
    model.eval()
    return model



# these ways of loading models are legacy code
def get_model(name, device=torch.device('cpu'), dataset='CIFAR10'):
    if type(name)==int:
        return from_database(name, dataset, device)
    if dataset=='CIFAR10':
        return get_model_CIFAR10(name, device=device)
    elif dataset=='CIFAR100':
        return get_model_CIFAR100(name, device=device)
    else:
        raise ValueError('No models registered for dataset ' + dataset)

        
def get_model_CIFAR10(name, device=torch.device('cpu')):
    folder = paths_config.project_folder + 'models/CIFAR10/'
    if name[:11]=='old_binary_':
        folder += 'binary/no_bias/'
        file = folder + name[4:] + '.pt'
        state_dict = torch.load(file, map_location='cpu')
        model = provable_classifiers.CNN_IBP(dset_in_name='CIFAR10', size=name[11:], last_bias=False, num_classes=1, last_layer_neg=False)
    elif name[:7]=='binary_':
        folder += 'binary/'
        file = folder + name + '.pt'
        state_dict = torch.load(file, map_location='cpu')
        model = provable_classifiers.CNN_IBP(dset_in_name='CIFAR10', size=name[7:], last_bias=True, num_classes=1, last_layer_neg=True)   
    elif name=='OE':
        file = folder + 'RN/CIFAR10_oe.pt'
        state_dict = torch.load(file, map_location='cpu')
        model = rn.get_ResNet()
    elif name=='hendrycks':
        file = folder + 'hendrycks_CIFAR10.pt'
        state_dict = torch.load(file, map_location='cpu')
        model = hendrycks.ResNet()
    elif name=='Plain':
        file = folder + 'RN/CIFAR10_plain.pt'
        state_dict = torch.load(file, map_location='cpu')
        model = rn.get_ResNet()
    elif name=='GOOD80':
        file = folder + 'binary/GOODQ80.pt'
        state_dict = torch.load(file, map_location='cpu')
        model = provable_classifiers.CNN_IBP(dset_in_name='CIFAR10', size='XL', last_bias=True)
    elif name=='GOOD100':
        file = folder + 'binary/GOODQ100.pt'
        state_dict = torch.load(file, map_location='cpu')
        model = provable_classifiers.CNN_IBP(dset_in_name='CIFAR10', size='XL', last_bias=True)
        
    elif name[:6]=='joint_':
        base_model = rn.get_ResNet()
        detector = get_model('binary_S')
        model = ibp.JointModel(base_model, detector)
        
        name = name[6:]
        folder += 'semi_joint/bias_shift/'
        file = folder + name + '.pt'
        state_dict = torch.load(file, map_location='cpu')
    else:
        raise ValueError(name + ' is not a valid model.')
        
    model.load_state_dict(state_dict) 
    model = model.to(device)
    model.eval()

    return model


def get_model_CIFAR100(name, device=torch.device('cpu')):
    folder = paths_config.project_folder + 'models/CIFAR100/'
    if name[:7]=='binary_':
        folder += 'binary/'
        size = name[7:]
        file = folder + 'CIFAR100_' + size + '.pt'
        state_dict = torch.load(file, map_location='cpu')
        model = provable_classifiers.CNN_IBP(dset_in_name='CIFAR100', size=size, last_bias=True, num_classes=1, last_layer_neg=True)   
    elif name=='Plain':
        file = folder + 'RN/CIFAR100_plain.pt'
        state_dict = torch.load(file, map_location='cpu')
        model = rn.get_ResNet(dset='CIFAR100')
    elif name=='OE':
        file = folder + 'RN/CIFAR100_oe.pt'
        state_dict = torch.load(file, map_location='cpu')
        model = rn.get_ResNet(dset='CIFAR100')
    elif name[:6]=='joint_':
        base_model = rn.get_ResNet(dset='CIFAR100')
        detector = get_model('binary_S', dataset='CIFAR100')
        model = ibp.JointModel(base_model, detector)
        
        name = name[6:]
        folder += 'semi_joint/bias_shift/'
        file = folder + name + '.pt'
        state_dict = torch.load(file, map_location='cpu')
    else:
        raise ValueError(name + ' is not a valid model.')
        
    model.load_state_dict(state_dict) 
    model = model.to(device)
    model.eval()

    return model



