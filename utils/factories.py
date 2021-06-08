import torch
import torch.nn as nn
import numpy as np

import utils.dataloaders.dataloading as dataloading
import utils.traintest.better_training as better 
import utils.models.provable_classifiers as provable_classifiers
import utils.models.resnet as rn
import utils.models.densenet as dn
import utils.traintest.schedules as schedules
import utils.traintest.losses as losses
import utils.model_zoo as zoo
import utils.models.modules_ibp as ibp
import utils.traintest.evaluators as evaluators
import paths_config
from utils.model_zoo import from_database


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_model(arch):
    if arch.arch_style.lower()=='cnn':
        model = provable_classifiers.CNN_IBP(dset_in_name=arch.dset_in_name, size=arch.arch_size, 
                                             last_bias=arch.use_last_bias, num_classes=arch.num_classes, 
                                             last_layer_neg=arch.last_layer_neg)
        if arch.last_layer_neg and arch.use_last_bias:
            with torch.no_grad():
                model.layers[-1].bias.data += 3.
    elif arch.arch_style.lower() in ['rn', 'resnet']:
        model = rn.get_ResNet(dset=arch.dset_in_name)
    elif arch.arch_style.lower()=='joint':
        base_model = rn.get_ResNet(dset=arch.dset_in_name)
        try:
            detector = torch.load(arch.detector_path)
        except:
            detector = zoo.get_model(arch.detector_path, dataset=arch.dset_in_name)
        with torch.no_grad():
            detector.layers[-1].bias.data += arch.bias_shift
        model = ibp.JointModel(base_model, detector, classes=arch.num_classes)
    elif arch.arch_style.lower()=='sep':
        base_model = from_database(arch.class_id, arch.dset_in_name)
        detector = from_database(arch.disc_id, arch.dset_in_name)
        detector.layers[-1].bias.data += arch.bias_shift
        model = ibp.JointModel(base_model, detector, classes=arch.num_classes)
    elif arch.arch_style.lower() in ['densenet']:
        model = dn.get_densenet(dataset=arch.dset_in_name, num_classes=arch.num_classes+1)
    elif arch.arch_style.lower() in ['dn']:
        model = dn.get_densenet(dataset=arch.dset_in_name, num_classes=arch.num_classes)
    else:
        raise Exception('Architecture not implemented.')
    
    if arch.file_path is not None:
        path = arch.file_path if arch.file_path[0]=='/' else paths_config.project_folder + arch.file_path
        try:
            state_dict = torch.load(path , map_location='cpu')
        except:
            idx = arch.file_path.find('ProvableOOD/') + 10
            state_dict = torch.load(paths_config.project_folder + path[idx:], map_location='cpu')
        try:
            model.load_state_dict(state_dict)
        except:
            print('WARNING: Attempting to load model despite incompatibility!')
            state_dict.pop('linear.bias')
            model.load_state_dict(state_dict)
    
    return model


def get_loaders(args):
    dataloader_kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': True}
    dataloader_kwargs_tiny = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    augmentation_train_in = {'crop': args.augmentation.crop, 'HFlip': args.augmentation.hflip, 'autoaugment': args.augmentation.autoaugment}
    
    train_loader_in = dataloading.datasets_dict[args.dset_in_name](train=True, batch_size=args.train.batch_size, 
                                                                   augmentation=augmentation_train_in, dataloader_kwargs=dataloader_kwargs)

    if args.dset_out_name == 'TINY':
        train_loader_out = dataloading.datasets_dict[args.dset_out_name](train=True, batch_size=args.train.batch_size, augmentation=augmentation_train_in, 
                                                                     dataloader_kwargs=dataloader_kwargs_tiny, shuffle=True,
                                                                         exclude_cifar=args.augmentation.train_exclude)
    else:
        train_loader_out = dataloading.datasets_dict[args.dset_out_name](train=True, batch_size=args.train.batch_size, augmentation=augmentation_train_in, 
                                                                     dataloader_kwargs=dataloader_kwargs) #exclude_cifar=args.augmentation.train_exclude causes problems if args.dset_out_name is not TINY
    
    
    test_loader_in, test_loaders_out = get_test_loaders(args)
    
    return train_loader_in, test_loader_in, train_loader_out, test_loaders_out


def get_test_loaders(args):
    dataloader_kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': True}

    test_loader_in = dataloading.datasets_dict[args.dset_in_name](train=False, batch_size=args.eval.batch_size, 
                                                                 augmentation=dict([]), dataloader_kwargs=dataloader_kwargs)
    
    test_loaders_out = {}
    val_dict = dataloading.val_loader_out_dicts[args.dset_in_name]
    for key in val_dict:
        loader = dataloading.datasets_dict[key](train=False, batch_size=args.eval.batch_size, augmentation=val_dict[key], 
                                              dataloader_kwargs=dataloader_kwargs)
        orig_size = len(loader.dataset)
        target_size = 400 if args.dset_in_name=='RImgNet' else 1000
        np.random.seed(seed=0)
        subset = np.random.choice(orig_size, size=min(orig_size, target_size), replace=False)
        sampler = torch.utils.data.SubsetRandomSampler(subset)
        dataloader_kwargs['sampler'] = sampler
        loader = dataloading.datasets_dict[key](train=False, batch_size=args.eval.batch_size, augmentation=val_dict[key], 
                                                      dataloader_kwargs=dataloader_kwargs)
        dataloader_kwargs['sampler'] = None
        test_loaders_out[key] = loader
    
    return test_loader_in, test_loaders_out


def get_evaluators(args):
    def fun(model, device, test_loader_in, test_loaders_out):
        from_logits = args.architecture.arch_style.lower()!='joint'
        IDEvaluator = evaluators.IDEvaluator(model, device, test_loader_in, from_logits=from_logits)
        
        if args.architecture.arch_style.lower() in ['joint', 'rn']:
            evaluatorOOD = evaluators.OODEvaluator(model, device, test_loader_in, test_loaders_out)
        else:
            evaluatorOOD = evaluators.BinaryEvaluator(model, device, test_loader_in, test_loaders_out)

        return [IDEvaluator, evaluatorOOD]
    return fun


def get_trainer_factory(train_args):
    schedule = schedules.schedule_dict[train_args.schedule.lr_schedule_type](train_args.schedule.lr_schedule, train_args.schedule.lr_schedule_epochs)
    kappa_schedule = schedules.schedule_dict[train_args.schedule.kappa_schedule_type](train_args.schedule.kappa_schedule, train_args.schedule.kappa_schedule_epochs)
    eps_schedule = schedules.schedule_dict[train_args.schedule.eps_schedule_type](train_args.schedule.eps_schedule, train_args.schedule.eps_schedule_epochs)
    
    in_loss = losses.loss_dict[train_args.in_loss_type]()
    out_loss = losses.loss_dict[train_args.out_loss_type]()
    if train_args.train_type=='GOOD':
        def fun(model, device, in_loader, out_loader, folder=None, writer=None, evaluators=[]):
            made_trainer = better.TrainerGOODER(model, device, in_loader, out_loader,
                                                schedule=schedule,
                                                folder=folder,
                                                writer=writer,
                                                evaluators=evaluators,
                                                epsilon=eps_schedule, 
                                                out_loss=out_loss,
                                                in_loss=in_loss,
                                                use_adam=train_args.use_adam,
                                                momentum=train_args.momentum,
                                                kappa_schedule=kappa_schedule,
                                                )

            return made_trainer
    elif train_args.train_type.upper()=='OE':
        def fun(model, device, in_loader, out_loader, folder=None, writer=None, evaluators=[]):
            made_trainer = better.TrainerOE(model, device, in_loader, out_loader,
                                            schedule=schedule,
                                            folder=folder,
                                            writer=writer,
                                            evaluators=evaluators,
                                            out_loss=out_loss,
                                            in_loss=in_loss,
                                            momentum=train_args.momentum,
                                            kappa_schedule=kappa_schedule
                                            )

            return made_trainer
        
    elif train_args.train_type.lower()=='plain':
        def fun(model, device, in_loader, out_loader, folder=None, writer=None, evaluators=[]):
            made_trainer = better.TrainerPlain(model, device, in_loader, out_loader,
                                            schedule=schedule,
                                            folder=folder,
                                            writer=writer,
                                            evaluators=evaluators,
                                            out_loss=out_loss,
                                            in_loss=in_loss,
                                            momentum=train_args.momentum,
                                            kappa_schedule=kappa_schedule
                                            )

            return made_trainer
    else:
        raise NotImplemented()
    return fun        
