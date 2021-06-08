# needs to import this in the beginning to work on SLURM
import skimage
from skimage import filters
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import torch
import argparse
from utils.factories import dotdict
import utils.eval_pipeline
import paths_config
from tinydb import TinyDB


parser = argparse.ArgumentParser(description='Define hyperparameters.', prefix_chars='-')

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default='CIFAR10', help='CIFAR10, CIFAR100, RImgNet')
parser.add_argument('--disc_id', type=int, required=True, help='doc_id of discriminator model within the database')
parser.add_argument('--class_id', type=int, required=True, help='doc_id of classifier model within the database')
parser.add_argument('--bias_shift', type=float, required=True, help='bias shift to use in combined model')

hps = parser.parse_args()
    
num_classes = {'CIFAR10':10, 'CIFAR100':100, 'RImgNet': 9}

def main():
    assert hps.dataset in ['CIFAR10', 'CIFAR100', 'RImgNet']
    path = paths_config.project_folder
    db = TinyDB(path + 'evals/' + hps.dataset + '.json')
    train_type = db.get(doc_id=hps.class_id)['args']['train']['train_type']
    
    batch_size = 25 if hps.dataset=='RImgNet' else 100
        
    name = 'Sep' + '_' + train_type + '_' + str(hps.bias_shift)
    args = {'architecture': {'arch_style': 'SEP', 'dset_in_name': hps.dataset, 
                             'disc_id': hps.disc_id, 'class_id': hps.class_id,
                             'bias_shift':hps.bias_shift, 'file_path': None, 'num_classes': num_classes[hps.dataset]},
            'eval': {'eps': 0.01, 'batch_size': batch_size},
            'dset_in_name': hps.dataset}
    data = {'name': name, 'args': args}
    
    doc_id = db.insert(data)

    args = dotdict({'dataset': hps.dataset, 'adversarial': True, 'guaranteed': True, 
                    'measures': ['AUC', 'AUPR', 'FPR@95'], 'eps': 0.01})
    device = torch.device('cuda:'+str(hps.gpu))
    utils.eval_pipeline.gen_eval(doc_id, args, device=device)
    
if __name__ == "__main__":
    main()
    