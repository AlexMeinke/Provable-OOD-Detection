import os

import torch
from torch import nn, optim
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils as vutils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import math
import json

import utils.traintest.logging as logging
import utils.traintest.schedules as schedules
import utils.traintest.losses as losses


class Trainer():
    def __init__(self, model, device, in_loader, out_loader, folder=None, writer=None, evaluators=[],
                 schedule=None, kappa_schedule=None, in_loss=nn.CrossEntropyLoss(), 
                 out_loss=losses.DirectLoss(), verbose=100, weight_decay=5e-4,
                 noise_refresh_interval_schedule=None, in_refresh_interval_schedule=None):
        self.model = model
        self.device = device
        self.in_loader = in_loader
        self.out_loader = out_loader
        self.evaluators = evaluators
        self.in_loss = in_loss
        self.out_loss = out_loss
        
        ####### Saving and Logging #######
        self.verbose = verbose
        self.writer = writer
        self.logged_scalars = {}
        self.logged_vectors = {}
        self.logged_img = {}
        self.logged_img_batch = {}
        
        self.folder = folder
        if self.folder is not None:
            if self.folder[-1] =='/':
                self.folder = self.folder[:-1]
            try:
                os.mkdir(self.folder)
            except FileExistsError:
                print(self.folder + ' already exists')
                
            try:
                os.mkdir(self.folder + '/checkpoints')
            except FileExistsError:
                print(self.folder + '/checkpoints already exists')
                
        self.logger = logging.Logger(file = self.folder + '/dump.pt')
        
        
        ####### Optimizer and lr schedules #######
        
        if schedule is None:
            self.lr_schedule = create_piecewise_constant_schedule([.1, .01, 0.001, 0.0001], [50, 75, 90, 100])
        else:
            self.lr_schedule = schedule
            
        self.kappa_schedule = 0.*self.lr_schedule if kappa_schedule is None else kappa_schedule
        self.kappa = kappa_schedule[0]
        
        self.epochs = self.lr_schedule.shape[0]
        self.weight_decay = weight_decay
        
        
    def train(self):
        self.model.train()
        self.model.to(self.device)
        
        out_loader = iter(self.out_loader)
        
        for epoch in range(self.epochs):            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr_schedule[epoch]
            
            self.inner_train(epoch, out_loader)
            self.save(epoch)
            with torch.no_grad():
                self.evaluate(epoch)
            self.log(epoch)
            
            
    def inner_train(self, epoch, out_loader):
        pass
    
    
    def save(self, epoch):
        print("Saving...")
        if self.folder is not None:
            if epoch == len(self.lr_schedule)-1:
                self.model.cpu()
                if type(self.model)==nn.DataParallel:
                    model = self.model.module
                else:
                    model = self.model
                torch.save(model.state_dict(), self.folder + '/final.pt')
                torch.save(model, self.folder + '/final.pth')
                self.model.to(self.device)
            elif epoch % 10 == 0:
                torch.save(self.model.state_dict(), self.folder + '/checkpoints/' + str(epoch) + '.pt')
                
                
    def log(self, epoch):
        print("Logging...")
        
        ### Log all registered Tensors
        if self.writer is not None:
            for key in self.logged_scalars:
                self.writer.add_scalar(key, self.logged_scalars[key], epoch)
            for key in self.logged_vectors:
                self.writer.add_histogram(key, self.logged_vectors[key], epoch)
            for key in self.logged_img:
                try:
                    self.writer.add_image(key, self.logged_img[key], epoch)
                except Exception as e:
                    print(f'Wtf is going on with {key} of shape {self.logged_img[key].shape}?')
                    print(e)
            for key in self.logged_img_batch:
                try:
                    self.writer.add_image(key, self.logged_img_batch[key], epoch, dataformats='NCHW')
                except Exception as e:
                    print(f'Wtf is going on with {key} of shape {self.logged_img_batch[key].shape}?')
                    print(e)
                    
        if epoch == len(self.lr_schedule)-1:
            for scalar in self.logged_scalars:
                try:
                    self.logged_scalars[scalar] = self.logged_scalars[scalar].item()
                except:
                    print('')
            eval_file = open(self.folder + "/results.json", "w")
            json.dump(self.logged_scalars, eval_file)
            eval_file.close()
            
        self.logger.clear()
        self.logged_scalars = {}
        self.logged_vectors = {}
        self.logged_img = {}
        self.logged_img_batch = {}
        
        
    def evaluate(self, epoch):
        print("Evaluating...")
        for evaluator in self.evaluators:
            evaluator.run(self.logged_scalars, self.logged_vectors, self.logged_img)

            
######## OE TRAINING ############

class TrainerOE(Trainer):
    def __init__(self, model, device, in_loader, out_loader, folder=None, writer=None, 
                 evaluators=[], schedule=None, kappa_schedule=None, in_loss=nn.CrossEntropyLoss(),
                 out_loss=losses.DirectLoss(), verbose=100, weight_decay=5e-4, momentum=0.9):
        
        super().__init__(model, device, in_loader, out_loader, folder, writer, evaluators, schedule,
                         kappa_schedule, in_loss, out_loss, verbose, weight_decay)
        
        if type(model)==nn.DataParallel:
            model = model.module
        parameters = model.base_model.parameters() if hasattr(model, 'base_model') else model.parameters()
        param_groups = [
                        {'params': parameters,'lr': self.lr_schedule[0], 'weight_decay': self.weight_decay},
                       ]
        
        self.optimizer = optim.SGD(param_groups, momentum=momentum, nesterov=True)
        
    def inner_train(self, epoch, out_loader):
        self.model.train()
        self.kappa = self.kappa_schedule[epoch]
        for batch_idx, (data, target) in enumerate(self.in_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            try:
                noise = next(out_loader)[0].to(self.device)
            except StopIteration:
                out_loader = iter(self.out_loader)
                noise = next(out_loader)[0].to(self.device)
                
            if data.shape[0]!=noise.shape[0]:
                continue
            
            full_data = torch.zeros_like( torch.cat([data, noise], 0) )
            full_data[::2] = data
            full_data[1::2] = noise
            output = self.model(full_data)
            out_in, out_out = output[::2], output[1::2]
            
            in_loss = self.in_loss(out_in, target)
            self.logger['in_loss'].append(in_loss)
            
            out_loss = self.out_loss(out_out)
            self.logger['out_loss'].append(out_loss)
            
            loss = in_loss.mean() + self.kappa*out_loss.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.logger['correct'].append(out_in.max(1)[1]==target)
            
            if (batch_idx % self.verbose == 0) and self.verbose>0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.in_loader.dataset),
                    100. * batch_idx / len(self.in_loader), loss.item()))
            
    def log(self, epoch):
        self.logger.concatenate()
        
        ### Log schedules
        self.logged_scalars['schedule/lr'] = self.optimizer.param_groups[0]['lr']
        self.logged_scalars['schedule/kappa'] = self.kappa
        

        self.logged_scalars['train/out_loss'] = self.logger['out_loss'].mean()
        
        self.logged_scalars['train/min_in_loss'] = self.logger['in_loss'].min()
        self.logged_scalars['train/max_in_loss'] = self.logger['in_loss'].max()
        self.logged_scalars['train/in_loss'] = self.logger['in_loss'].mean()
        self.logged_scalars['train/acc'] = self.logger['correct'].float().mean()
        
        super().log(epoch)
        

class TrainerPlain(Trainer):
    def __init__(self, model, device, in_loader, out_loader, folder=None, writer=None, 
                 evaluators=[], schedule=None, kappa_schedule=None, in_loss=nn.CrossEntropyLoss(),
                 out_loss=losses.DirectLoss(), verbose=100, weight_decay=5e-4, momentum=0.9):
        
        super().__init__(model, device, in_loader, out_loader, folder, writer, evaluators, schedule,
                         kappa_schedule, in_loss, out_loss, verbose, weight_decay)
        
        if type(model)==nn.DataParallel:
            model = model.module
        parameters = model.base_model.parameters() if hasattr(model, 'base_model') else model.parameters()
        param_groups = [
                        {'params': parameters,'lr': self.lr_schedule[0], 'weight_decay': self.weight_decay},
                       ]
        
        self.optimizer = optim.SGD(param_groups, momentum=momentum)
        
    def inner_train(self, epoch, out_loader):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.in_loader):
            data, target = data.to(self.device), target.to(self.device)
            out_in = self.model(data)
            
            in_loss = self.in_loss(out_in, target)
            self.logger['in_loss'].append(in_loss)
            
            loss = in_loss.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.logger['correct'].append(out_in.max(1)[1]==target)
            
            if (batch_idx % self.verbose == 0) and self.verbose>0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.in_loader.dataset),
                    100. * batch_idx / len(self.in_loader), loss.item()))
            
    def log(self, epoch):
        self.logger.concatenate()
        
        ### Log schedules
        self.logged_scalars['schedule/lr'] = self.optimizer.param_groups[0]['lr']
        self.logged_scalars['schedule/kappa'] = self.kappa
        
        self.logged_scalars['train/min_in_loss'] = self.logger['in_loss'].min()
        self.logged_scalars['train/max_in_loss'] = self.logger['in_loss'].max()
        self.logged_scalars['train/in_loss'] = self.logger['in_loss'].mean()
        self.logged_scalars['train/acc'] = self.logger['correct'].float().mean()
        
        super().log(epoch)


######## GOODER TRAINING ############

class TrainerGOODER(Trainer):
    def __init__(self, model, device, in_loader, out_loader, folder=None, writer=None, 
                 evaluators=[], schedule=None, kappa_schedule=None, in_loss=nn.CrossEntropyLoss(),
                 out_loss=losses.DirectLoss(), verbose=100, weight_decay=5e-4, 
                 epsilon=0.01, use_adam=False, momentum=0.9):
        
        super().__init__(model, device, in_loader, out_loader, folder, writer, evaluators, schedule,
                         kappa_schedule, in_loss, out_loss, verbose, weight_decay)
        
        
        ####### Optimizer #######
        if type(model)==nn.DataParallel:
            model = model.module
        last_layer_ids = [id(p) for p in model.layers[-1].parameters()]
        non_last_layer_params = filter(lambda p: id(p) not in last_layer_ids, model.parameters())
        
        param_groups = [{'params': non_last_layer_params, 'lr': self.lr_schedule[0], 'weight_decay': self.weight_decay},
                        {'params': model.layers[-1].parameters(),'lr': self.lr_schedule[0], 'weight_decay': 0.},
                       ]
            
        if use_adam:
            self.optimizer = optim.Adam(param_groups)
        else:
            self.optimizer = optim.SGD(param_groups, momentum=momentum)
        
        
        ####### IBP #######        
        if epsilon is None:
            self.eps_schedule = 0.01 + 0.*self.lr_schedule
        elif not hasattr(epsilon, '__iter__'):
            self.eps_schedule = epsilon + 0.*self.lr_schedule
        else:
            self.eps_schedule = epsilon
        self.epsilon = self.eps_schedule[0]
        
        
    def inner_train(self, epoch, out_loader):
        before = time.time()        
        
        self.kappa = self.kappa_schedule[epoch]
        self.epsilon = self.eps_schedule[epoch]
        

        for batch_idx, (data, target) in enumerate(self.in_loader):
            self.model.train()
            if batch_idx<5:
                self.logger['data'].append(data)
                self.logger['target'].append(target)
            
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            self.logger['output'].append(output)

            loss = self.in_loss(output).mean()
            
            # only load out-dist data if necessary
            if self.kappa > 0.:
                try:
                    noise = next(out_loader)[0].to(self.device)
                except StopIteration:
                    out_loader = iter(self.out_loader)
                    noise = next(out_loader)[0].to(self.device)
                    
                if batch_idx<5:
                    self.logger['noise'].append(noise)
                self.model.eval()
 
            # OOD loss term
            if self.kappa > 0.:
                lb = torch.clamp(noise - self.epsilon, 0, 1)
                ub = torch.clamp(noise + self.epsilon, 0, 1)
                if type(self.model)==nn.DataParallel:
                    l, u = self.model.module.ibp_forward(lb, ub)
                else:
                    l, u = self.model.ibp_forward(lb, ub)
                ub_log_conf_out_batch = u.max(dim=-1)[0]
                
                out_loss = self.kappa * self.out_loss(-ub_log_conf_out_batch)
                self.logger['out_losses'].append(out_loss)
                loss += out_loss.mean()
            
            self.optimizer.zero_grad()
            
            try:
                loss.backward()
            except RuntimeError as inst: # can only trigger if option 'detect_anomaly' is true
                self.logger.dump('dump' + str(epoch) + '.pt')
                raise inst
            
            # log effective gradients
#             prev_weights = [l.weight.data.clone() if hasattr(l, 'weight') else None for l in self.model.layers]
            self.optimizer.step()
            
                
#             with torch.no_grad():
#                 for i, l in enumerate(self.model.layers):
#                     if hasattr(l, 'weight'):
#                         prev_weight = prev_weights[i]
#                         weight = l.weight.data
#                         eff_grad = weight - prev_weight
#                         self.logger['eff_grad_' + str(i)].append( eff_grad.norm().unsqueeze(0) )
#                         self.logger['eff_rel_grad_' + str(i)].append( (eff_grad / (weight + 1e-8) ).norm().unsqueeze(0) )
            
            ####### Console Logging #######
            self.logger['train_losses'].append(loss)
            
            if (batch_idx % self.verbose == 0) and self.verbose>0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.in_loader.dataset),
                    100. * batch_idx / len(self.in_loader), loss.item()))
            
        ####### TB Logging #######
        self.logged_scalars['time/total_time'] = time.time() - before
                
                
    def log(self, epoch):
        self.logger.concatenate()
        
        ### Log schedules
        self.logged_scalars['schedule/lr'] = self.optimizer.param_groups[0]['lr']
        self.logged_scalars['schedule/kappa'] = self.kappa
        self.logged_scalars['schedule/epsilon'] = self.epsilon
        
#         ### Log weights and Gradients
#         ratios = len(self.model.layers)*[None]
#         for i, l in enumerate(self.model.layers):
#             if hasattr(l, 'weight'):
#                 weight = l.weight.data.cpu()
#                 grad = l.weight.grad
#                 self.logged_scalars['weights/layer' + str(i) + '_l1'] = weight.abs().mean()
#                 self.logged_scalars['weights/layer' + str(i) + '_mean'] = weight.mean()
#                 self.logged_scalars['weights/layer' + str(i) + '_var'] = weight.var()
                
#                 if grad is not None:
#                     ratios[i] = (grad.cpu() / (weight + 1e-8)).abs()
#                 else:
#                     ratios[i] = torch.zeros_like(weight)
                    
#                 self.logged_scalars['weight_gradients/layer' + str(i) + '_median'] = ratios[i].median()
#                 self.logged_scalars['weight_gradients_mean/layer' + str(i) + '_mean'] = ratios[i].mean()
#                 self.logged_scalars['effective_gradients/layer' + str(i) + '_mean'] = self.logger['eff_grad_' + str(i)].mean()
#                 self.logged_scalars['effective_relative_gradients/layer' + str(i) + '_mean'] = self.logger['eff_rel_grad_' + str(i)].mean()
            
        if len(self.logger['out_losses'])>0:
            self.logged_scalars['train/out_loss'] = self.logger['out_losses'].mean()
        
        self.logged_scalars['train/min_train_loss'] = self.logger['train_losses'].min()
        self.logged_scalars['train/max_train_loss'] = self.logger['train_losses'].max()
        self.logged_scalars['train/train_loss'] = self.logger['train_losses'].mean()
        
        self.logged_scalars['train/train_acc'] = self.logger['correct'].mean()

        super().log(epoch)
