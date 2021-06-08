import yaml
import torch
from tinydb import TinyDB
import utils.traintest.evaluation as ev
import utils.dataloaders.dataloading as dataloading
import utils.factories as fac
import utils.adversarial.aauc as aauc
import paths_config
from utils.factories import dotdict
import numpy as np


measure_dict = {'AUC':ev.auroc_conservative, 'AUPR':ev.auprc, 'FPR@95':lambda o1, o2: ev.fpr_at_tpr(o1, o2, .95)}
# measure_dict = {'AUC':ev.auroc_conservative, 'FPR@95':lambda o1, o2: ev.fpr_at_tpr(o1, o2, .95)}


def gen_eval(doc_id, args, device=torch.device('cuda:0')):
    ##### Access database #####
    db = TinyDB(paths_config.project_folder + 'evals/' + args.dataset + '.json')
    data = {}
    
    entry = db.get(doc_id=doc_id)
    train_args = dotdict(entry['args'])
    for sub in train_args:
        if type(train_args[sub])==dict:
            train_args[sub] = dotdict(train_args[sub])
    
    
    ##### Get Loaders #####
    if 'batch_size' in train_args['eval']:
        batch_size = train_args['eval']['batch_size']
    else:
        batch_size = 100
    train_args['eval']['batch_size'] = batch_size
    
    in_loader, loaders_out = fac.get_test_loaders(train_args)
    
    
    ##### Get Model #####
    model = fac.get_model(dotdict(train_args['architecture'])).to(device)
    model.eval()
    arch_style = train_args['architecture']['arch_style']
    if arch_style.lower()=='sep':
        arch_style = 'joint'
    
    
    ##### ID Stats #####
    num_classes = train_args['architecture']['num_classes']
    from_logits = (arch_style.lower() in ['rn', 'cnn', 'dn']) and num_classes!=1
    use_last_class = arch_style.lower()=='densenet'
    
    if num_classes!=1:
        acc, conf = ev.get_accuracy(model, device, in_loader, from_logits=from_logits)
        data['acc'] = acc
        data['MMC'] = conf.mean().item()
        if use_last_class:
            conf = - torch.log_softmax(ev.get_output(model, device, in_loader), dim=1)[:,-1]
    else:
        conf = torch.sigmoid( ev.get_output(model, device, in_loader).squeeze() )
    
    
    ##### OOD Stats #####
    data['OOD'] = {}
    
    for measure in args.measures:
        data['OOD'][measure] = {}
        data['OOD']['G'+measure] = {}
        data['OOD']['A'+measure] = {}
    
    for out_name in loaders_out:
        print(out_name)
        loader = loaders_out[out_name]
        if num_classes!=1:
            if use_last_class:
                conf_out = - torch.log_softmax(ev.get_output(model, device, loader), dim=1)[:,-1]
            else:
                conf_out = ev.get_conf(model, device, loader, from_logits=from_logits)
        else:
            conf_out = torch.sigmoid(ev.get_output(model, device, loader).squeeze())
        
        if args.guaranteed:
            if arch_style.lower() in ['rn', 'dn']:
                conf_g = torch.tensor([1.])
            elif arch_style.lower() in ['densenet']:
                conf_g = torch.tensor([1e6])
            elif arch_style=='CNN' and num_classes==1:
                conf_g = torch.sigmoid( ev.get_output_ub(model, device, loader, eps=args.eps) ).squeeze()
            elif arch_style=='CNN' and num_classes!=1:
                conf_g = np.exp(ev.get_ub_log_conf(model, device, loader, eps=args.eps))
            elif arch_style=='joint':
                conf_g = ev.get_output_ub(model, device, loader, eps=args.eps).exp()
                
        if args.adversarial:
            batches = 40 if args.dataset=='RImgNet' else 10
            attack_args = dotdict({'batches': batches, 'num_classes': num_classes, 
                                   'eps': args.eps, 'from_logits': from_logits, 
                                   'iterations': 200, 'dataset': args.dataset, 
                                   'use_last_class': use_last_class, 'save_loc': None})
            conf_a = aauc.get_conf_lb(model, device, loader, attack_args).squeeze()
        
        for measure in args.measures:
            data['OOD'][measure][out_name] = measure_dict[measure](conf, conf_out)
            
            if args.guaranteed:
                data['OOD']['G'+measure][out_name] = measure_dict[measure](conf, conf_g)
            if args.adversarial:
                data['OOD']['A'+measure][out_name] = measure_dict[measure](conf, conf_a)
                
    db.update({'results': data}, doc_ids=[doc_id])
    return data