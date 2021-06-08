# needs to import this in the beginning to work on SLURM
import skimage
from skimage import filters
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import torch
import torch.nn as nn
import os
import datetime
import hydra
import sys
from contextlib import redirect_stdout
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
import omegaconf
import yaml
import operator
from tinydb import TinyDB, Query
import paths_config
import utils.factories as fac
import utils.eval_pipeline as eval_pipeline
from utils.factories import dotdict


@hydra.main(config_path='prood_config', config_name='default')
def main(args):
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    folder = os.getcwd() # hydra automatically creates run-specific folders
    time_name = '|'.join(folder.split('/')[-2:])
    with open('train_ProoD.log', 'w') as f:
        with redirect_stdout(f):
            tb_folder = get_tb_folder(args, time_name)
            print(tb_folder)
            writer = SummaryWriter(tb_folder)
            config = yaml.load(open('.hydra/config.yaml', 'r'), Loader=yaml.FullLoader)
            config_str = pretty_dump_yaml(config)
            print(pretty_dump_yaml(config, 0, sep=' '))
            writer.add_text('Hyperparameters', config_str)
            
            model = fac.get_model(args.architecture)
            if type(args.gpu)==omegaconf.listconfig.ListConfig:
                device = torch.device('cuda:' + str(min(args.gpu)))
                model = nn.DataParallel(model, device_ids=args.gpu)
                eval_model = model.module
            else:
                device = torch.device('cuda:' + str(args.gpu))
                eval_model = model
            model = model.to(device)
            eval_model = eval_model.to(device)

            train_loader_in, test_loader_in, train_loader_out, test_loaders_out = fac.get_loaders(args)
            
            evaluator_list = fac.get_evaluators(args)(eval_model, device, test_loader_in, test_loaders_out)

            trainer_factory = fac.get_trainer_factory(args.train)
            trainer = trainer_factory(model, device, train_loader_in, train_loader_out, folder, writer, evaluators=evaluator_list)

            trainer.train()
            
            doc_id = register_model(folder)
            test_args = dotdict({'dataset': args.dset_in_name, 'adversarial': True, 'guaranteed': True, 
                                 'measures': ['AUC', 'AUPR', 'FPR@95'], 'eps': args.eval.eps})
            if type(args.gpu)!=omegaconf.listconfig.ListConfig: # don't waste multi-gpu resources
                eval_pipeline.gen_eval(doc_id, test_args, device=device)


def get_tb_folder(args, time_name):
    base_folder = paths_config.project_folder
    base_folder += 'tb_logs/' + args.architecture.arch_style + '/'
    tb_folder = base_folder + time_name + '|'
    if args.tb_name.string_base is not None:
        tb_folder += args.tb_name.string_base
    if args.tb_name.string_extra is not None:
        tb_folder += args.tb_name.string_extra
    if args.tb_name.hp_base is not None:
        for key, item in args.tb_name.hp_base.items():
            tb_folder += '|' + key + '='
            tb_folder += get_tb_str(operator.attrgetter(item)(args))
    if args.tb_name.hp_extra is not None:
        for key, item in args.tb_name.hp_extra.items():
            tb_folder += '|' + key + '='
            tb_folder += get_tb_str(operator.attrgetter(item)(args))
    return hydra.utils.to_absolute_path(tb_folder)


def get_tb_str(a):
    if OmegaConf.is_list(a):
        x = a[-1]
    else:
        x = a
    if str(x) == 'True':
        return 'T'
    if str(x) == 'False':
        return 'F'
    else:
        return str(x)


def pretty_dump_yaml(y, indent=0, sep='&nbsp;'):
    full_string = ''
    prefix = 2*indent*sep
    for name in y:
        item = y[name]
        full_string += prefix + name + ': '
        if type(item) is dict:
            full_string += '  \n'
            full_string += pretty_dump_yaml(item, indent=indent+1)
            full_string += '  \n'
        else:
            full_string += str(y[name]) + '  \n'
            
    return full_string


def register_model(folder):
    with open(folder + '/.hydra/config.yaml', 'r') as stream:
        args = yaml.safe_load(stream)
        
    args['architecture']['file_path'] = folder + '/final.pt'
    name = args['architecture']['arch_style'] + '_' + args['train']['train_type']
    data = {'name': name, 'args': args}
    
    path = paths_config.project_folder
    db = TinyDB(path + 'evals/' + args['dset_in_name'] + '.json')
    return db.insert(data)


if __name__ == "__main__":
    main()
    