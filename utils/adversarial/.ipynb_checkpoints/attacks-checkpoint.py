import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import math
import torch.optim as optim
# from utils.train_test.losses import *

import utils.adversarial.auto_attack.autopgd_pt as apgd


class MaxConf(nn.Module):
    def __init__(self, from_logits=True):
        super().__init__()
        self.from_logits = from_logits
        
    def forward(self, x, y, x1, y1, reduction='mean'):
        if self.from_logits:
            out = torch.softmax(y, dim=1)
        else:
            out = y
        out = -out.max(1)[0]
        
        if reduction=='mean':
            return out.mean()
        elif reduction=='none':
            return out
        else:
            print('Error, reduction unknown!')
            

class LastConf(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, x1, y1, reduction='mean'):
        out = torch.log_softmax(y, dim=1)
        out = out[:,-1]
        
        if reduction=='mean':
            return out.mean()
        elif reduction=='none':
            return out
        else:
            print('Error, reduction unknown!')
            
            
class MaxConfDouble(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, x1, y1, reduction='mean'):
        y_double = y.double()
        out = torch.log_softmax(y_double, dim=1)
        out = -out.max(1)[0]
        
        if reduction=='mean':
            return out.mean()
        elif reduction=='none':
            return out
        else:
            print('Error, reduction unknown!')
            
            
class MinConf(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, x1, y1, reduction='mean'):

        out = torch.softmax(y, dim=1)
        out = out.max(1)[0]
        
        if reduction=='mean':
            return out.mean()
        elif reduction=='none':
            return out
        else:
            print('Error, reduction unknown!')
            
            
class MaxConfSpec(nn.Module):
    def __init__(self, y):
        super().__init__()
        self.cls = y
        
    def forward(self, x, y, x1, y1, reduction='mean'):

        out = torch.softmax(y, dim=1)
        out = -out[:,self.cls]
        
        if reduction=='mean':
            return out.mean()
        elif reduction=='none':
            return out
        else:
            print('Error, reduction unknown!')
            
            
class MaxConfCCU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, x1, y1, reduction='mean'):
        out = -y.max(1)[0]
        
        if reduction=='mean':
            return out.mean()
        elif reduction=='none':
            return out
        else:
            print('Error, reduction unknown!')

            
            
def normalize_perturbation(perturbation, p):
    if p == 'inf':
        return perturbation.sign()
    elif p==2 or p==2.0:
        bs = perturbation.shape[0]
        pert_flat = perturbation.view(bs, -1)
        pert_normalized = torch.nn.functional.normalize(pert_flat, p=p, dim=1)
        return pert_normalized.view_as(perturbation)
    else:
        raise NotImplementedError('Projection only supports l2 and inf norm')


def project_perturbation(perturbation, eps, p):
    if p == 'inf':
        mask = perturbation.abs() > eps
        pert_normalized = perturbation
        pert_normalized[mask] = eps * perturbation[mask].sign()
        return pert_normalized
    elif p==2 or p==2.0:
        #TODO use torch.renorm
        bs = perturbation.shape[0]
        pert_flat = perturbation.view(bs, -1)
        norm = torch.norm(perturbation.view(bs, -1), dim=1) + 1e-10
        mask = norm > eps
        pert_normalized = pert_flat
        pert_normalized[mask, :] = (eps / norm[mask, None]) * pert_flat[mask, :]
        return pert_normalized.view_as(perturbation)
    else:
        raise NotImplementedError('Projection only supports l2 and inf norm')


###################################################
class AdversarialNoiseGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, x):
        #generate noise matching the size of x
        raise NotImplementedError()

        
class Contraster(AdversarialNoiseGenerator):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        eps = self.eps
        s = (x>(1-eps)).float() + torch.clamp(x * (x<=(1-eps)).float() -eps, 0, 1)
        return s-x
    
    
class DeContraster(AdversarialNoiseGenerator):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        diff = torch.clamp(x.mean(dim=(1,2,3))[:,None,None,None] - x, -self.eps, self.eps)
        return diff
    
    
class Lowerer(AdversarialNoiseGenerator):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        eps = self.eps
        s = torch.clamp(x-eps, 0, 1)
        return s-x
    
    
class Replacer(AdversarialNoiseGenerator):
    def __init__(self, preloaded):
        super().__init__()
        self.preloaded = preloaded
        self.i = 0

    def forward(self, x):
        num = x.shape[0]
        s = self.preloaded[self.i:self.i+num].to(x.device)
        self.i += num
        return s-x
    
    
# class Smoother(AdversarialNoiseGenerator):
#     def __init__(self, sigma=0.01):
#         super().__init__()
#         self.sigma = sigma
#         self.i = 0

#     def forward(self, x):
#         s = self.filter_gauss(x, srange=[1,2.5])
#         return s-x

#     def filter_gauss(datapoint, srange=[1,1]):
#         img, label = datapoint
#         imgn = np.transpose(img.numpy(), (1,2,0))
#         sigma = srange[0] + np.random.random_sample()*(srange[1]-srange[0])
#         imgn_gaussed = skimage.filters.gaussian(imgn, sigma=sigma, multichannel=3)
#         return torch.from_numpy(np.transpose(imgn_gaussed, (2,0,1))), label #+ ('gauss', sigma)       
    
        
class UniformNoiseGenerator(AdversarialNoiseGenerator):
    def __init__(self, min=0.0, max=1.0):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return (self.max - self.min) * torch.rand_like(x) + self.min

    
class NormalNoiseGenerator(AdversarialNoiseGenerator):
    def __init__(self, sigma=1.0, mu=0):
        super().__init__()
        self.sigma = sigma
        self.mu = mu

    def forward(self, x):
        return self.sigma * torch.randn_like(x) + self.mu

    
class CALNoiseGenerator(AdversarialNoiseGenerator):
    def __init__(self, rho=1, lambda_scheme='normal'):
        super().__init__()
        self.rho = rho
        self.lambda_scheme = lambda_scheme

    def forward(self, x):
        if self.lambda_scheme == 'normal':
            lambda_targets =  x.new_zeros(x.shape[0])
            reject_idcs = lambda_targets < 1
            #rejection sample from truncated normal
            while sum(reject_idcs > 0):
                lambda_targets[reject_idcs] = math.sqrt(self.rho) * torch.randn(sum(reject_idcs), device=x.device).abs() + 1e-8
                reject_idcs =  lambda_targets > 1
        elif self.lambda_scheme == 'uniform':
            lambda_targets = torch.rand(x.shape[0], device=x.device)

        target_dists_sqr = -torch.log( lambda_targets) * self.rho
        dirs = torch.randn_like(x)
        dirs_lengths = torch.norm( dirs.view( x.shape[0], -1)  , dim=1)
        dirs_normalized = dirs / dirs_lengths.view(x.shape[0], 1, 1, 1)
        perts = target_dists_sqr.sqrt().view(x.shape[0], 1, 1, 1) * dirs_normalized
        return perts


#############################################iterative PGD attack
def logits_diff_loss(out, y_oh, reduction='mean'):
    #out: model output
    #y_oh: targets in one hot encoding
    #confidence:
    out_real = torch.sum((out * y_oh), 1)
    out_other = torch.max(out * (1. - y_oh) - y_oh * 100000000., 1)[0]

    diff = out_other - out_real

    return TrainLoss.reduce(diff, reduction)


class Adversarial_attack():
    def __init__(self, loss, num_classes, model=None, save_trajectory=False):
        #loss should either be a string specifying one of the predefined loss functions
        #OR
        #a custom loss function taking 4 arguments as train_loss class
        self.loss = loss
        self.save_trajectory = False
        self.last_trajectory = None
        self.num_classes = num_classes
        if model is not None:
            self.model = model
        else:
            self.model = None

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)

    def set_loss(self, loss):
        self.loss = loss

    def _get_loss_f(self, x, y, targeted, reduction):
        #x, y original data / target
        #targeted whether to use a targeted attack or not
        #reduction: reduction to use: 'sum', 'mean', 'none'
        if isinstance(self.loss, str):
            if self.loss.lower() =='crossentropy':
                if not targeted:
                    l_f = lambda data, data_out: -torch.nn.functional.cross_entropy(data_out, y, reduction=reduction)
                else:
                    l_f = lambda data, data_out: torch.nn.functional.cross_entropy(data_out, y, reduction=reduction )
            elif self.loss.lower() == 'logitsdiff':
                if not targeted:
                    y_oh = torch.nn.functional.one_hot(y, self.num_classes)
                    y_oh = y_oh.float()
                    l_f = lambda data, data_out: -logits_diff_loss(data_out, y_oh, reduction=reduction)
                else:
                    y_oh = torch.nn.functional.one_hot(y, self.num_classes)
                    y_oh = y_oh.float()
                    l_f = lambda data, data_out: logits_diff_loss(data_out, y_oh, reduction=reduction)
            else:
                raise ValueError(f'Loss {self.loss} not supported')
        else:
            #for monotone pgd, this has to be per patch example, not mean
            l_f = lambda data, data_out: self.loss(data, data_out, x, y, reduction=reduction)

        return l_f

    def get_config_dict(self):
        raise NotImplementedError()

    def get_last_trajectory(self):
        if not self.save_trajectory or self.last_trajectory is None:
            raise AssertionError()
        else:
            return self.last_trajectory

    def __get_trajectory_depth(self):
        raise NotImplementedError()

    def set_model(self, model):
        self.model = model

    def check_model(self):
        if self.model is None:
            raise RuntimeError('Attack model not set')

    def perturb(self, x, y, targeted=False):
        #force child class implementation
        raise NotImplementedError()

        
class Restart_attack(Adversarial_attack):
    #Base class for attacks that start from different initial values
    #Make sure that they MINIMIZE the given loss function
    def __init__(self, loss, restarts,  num_classes, model=None, save_trajectory=False):
        super().__init__(loss, num_classes, model=model, save_trajectory=save_trajectory)
        self.restarts = restarts

    def perturb_inner(self, x, y, targeted=False):
        #force child class implementation
        raise NotImplementedError()

    def perturb(self, x, y, targeted=False):
        #base class method that handles various restarts
        self.check_model()

        is_train = self.model.training
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False


        restarts_data = x.new_empty((1 + self.restarts,) + x.shape)
        restarts_objs = x.new_empty((1 + self.restarts, x.shape[0]))

        if self.save_trajectory:
            self.last_trajectory = None
            trajectories_shape = (self.restarts,) + (self.__get_trajectory_depth(),) + x.shape
            restart_trajectories = x.new_empty(trajectories_shape)

        for k in range(1 + self.restarts):
            k_data, k_obj, k_trajectory = self.perturb_inner(x, y, targeted=targeted)
            restarts_data[k, :] = k_data
            restarts_objs[k, :] = k_obj
            if self.save_trajectory:
                restart_trajectories[k, :] = k_trajectory

        bs = x.shape[0]
        best_idx = torch.argmin(restarts_objs, 0)
        best_data = restarts_data[best_idx, range(bs), :]

        if self.save_trajectory:
            self.last_trajectory = restart_trajectories[best_idx, :, range(bs), :]

        #reset model status
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = True

        return best_data

    
class DummyAttack(Adversarial_attack):
    def __init__(self):
        super().__init__(None, 0, model=None)

    def perturb(self, x, y, targeted=False):
        return x
    
        
class APGD(Adversarial_attack):
    def __init__(self, eps, device, model=None, n_iter=100):
        super().__init__(None, 0, model=model)
        self.attack = apgd.APGDAttack(model, device=device, eps=eps, n_iter=n_iter)
        
    def perturb(self, x, y, targeted=False):
        _, adv_sample = self.attack.perturb(x, y)
        return adv_sample
    

class FGM(Restart_attack):
    #one step attack with l2 or inf norm constraint
    def __init__(self, eps, num_classes, norm='inf', loss='CrossEntropy', restarts=0, 
                 init_noise_generator=None, model=None, save_trajectory=False):
        super().__init__(loss, restarts, num_classes, model=model, save_trajectory=save_trajectory)
        self.eps = eps
        self.norm = norm
        self.init_noise_generator = init_noise_generator

    def __get_trajectory_depth(self):
        return 2

    def get_config_dict(self):
        dict = {}
        dict['type'] = 'FGM'
        dict['eps'] = self.eps
        dict['norm'] = self.norm
        dict['restarts'] = self.restarts
        #dict['init_sigma'] = self.init_sigma
        return dict

    def perturb_inner(self, x, y, targeted=False):
        l_f = self._get_loss_f(x, y, targeted, 'none')


        if self.init_noise_generator is None:
            pert = torch.zeros_like(x)
        else:
            pert = self.init_noise_generator(x)
            pert = torch.clamp(x + pert, 0, 1) - x  # box constraint
            pert = project_perturbation(pert, self.eps, self.norm)

        pert.requires_grad_(True)

        with torch.enable_grad():
            p_data = x + pert
            out = self.model(p_data)
            loss_expanded = l_f(p_data, out)
            loss = loss_expanded.mean()
            grad = torch.autograd.grad(loss, pert)[0]

        with torch.no_grad():
            pert = project_perturbation(pert - grad, self.eps, self.norm)
            p_data = x + pert
            p_data = torch.clamp(p_data, 0, 1)

        if self.save_trajectory:
            trajectory = torch.zeros((2,) + x.shape, device=x.device)
            trajectory[0, :] = x
            trajectory[1, :] = p_data
        else:
            trajectory = None

        return p_data, l_f(self.model(p_data)), trajectory

    
def create_early_stopping_mask(out, y, conf_threshold, targeted):
    finished = False
    conf, pred = torch.max(torch.nn.functional.softmax(out, dim=1), 1)
    conf_mask = conf > conf_threshold
    if targeted:
        correct_mask = torch.eq(y, pred)
    else:
        correct_mask = (~torch.eq(y, pred))

    mask = 1. - (conf_mask & correct_mask).float()

    if sum(1.0 - mask) == out.shape[0]:
        finished = True

    mask = mask[(..., ) + (None, ) * 3]
    return finished, mask


class PGD(Restart_attack):
    def __init__(self, eps, iterations, stepsize, num_classes, momentum=0.9, decay=1.0, norm='inf', loss='CrossEntropy',
                 normalize_grad=False, early_stopping=0, restarts=0, init_noise_generator=None, model=None,
                 save_trajectory=False):
        super().__init__(loss, restarts, num_classes, model=model, save_trajectory=save_trajectory)
        #loss either pass 'CrossEntropy' or 'LogitsDiff' or custom loss function
        self.eps = eps
        self.iterations = iterations
        self.stepsize = stepsize
        self.momentum = momentum
        self.decay = decay
        self.norm = norm
        self.loss = loss
        self.normalize_grad = normalize_grad
        self.early_stopping = early_stopping
        self.init_noise_generator = init_noise_generator

    def __get_trajectory_depth(self):
        return self.iterations + 1

    def get_config_dict(self):
        dict = {}
        dict['type'] = 'PGD'
        dict['eps'] = self.eps
        dict['iterations'] = self.iterations
        dict['stepsize'] = self.stepsize
        dict['norm'] = self.norm
        if isinstance(self.loss, str):
            dict['loss'] = self.loss
        dict['restarts'] = self.restarts
        #dict['init_sigma'] = self.init_sigma
        return dict


    def perturb_inner(self, x, y, targeted=False):
        l_f = self._get_loss_f(x, y, targeted, 'none')

        velocity = torch.zeros_like(x)

        #initialize perturbation
        if self.init_noise_generator is None:
            pert = torch.zeros_like(x)
        else:
            pert = self.init_noise_generator(x)

        #trajectory container
        if self.save_trajectory:
            trajectory = torch.zeros((self.iterations + 1,) + x.shape, device=x.device)
            trajectory[0, :] = x
        else:
            trajectory = None

        for i in range(self.iterations):
            pert.requires_grad_(True)
            with torch.enable_grad():
                p_data = x + pert
                out = self.model(p_data)

                if self.early_stopping > 0:
                    finished, mask = create_early_stopping_mask(out, y, self.early_stopping, targeted)
                    if finished:
                        break
                else:
                    mask = 1.

                loss_expanded = l_f(p_data, out)
                loss = loss_expanded.mean()
                grad = torch.autograd.grad(loss, pert)[0]

            with torch.no_grad():

                # pgd on given loss
                if self.normalize_grad:
                    # https://arxiv.org/pdf/1710.06081.pdf the l1 normalization follows the momentum iterative method
                    l1_norm_gradient =  1e-10 + torch.sum(grad.abs().view(x.shape[0], -1), dim=1).view(-1,1,1,1)
                    velocity = self.momentum * velocity + grad / l1_norm_gradient
                    norm_velocity = normalize_perturbation(velocity, self.norm)
                else:
                    # velocity update as in pytorch https://pytorch.org/docs/stable/optim.html
                    velocity = self.momentum * velocity + grad
                    norm_velocity = velocity

                pert = pert - (self.decay**i) * self.stepsize * mask * norm_velocity
                pert = project_perturbation(pert, self.eps, self.norm)
                pert = torch.clamp(x + pert, 0, 1) - x #box constraint

                if self.save_trajectory:
                    trajectory[i + 1] = x + pert

        p_data = x + pert
        return p_data, l_f(p_data, self.model(p_data)), trajectory

    
###################################################################################################
class ArgminPGD(Restart_attack):
    def __init__(self, eps, iterations, stepsize, num_classes, momentum=0.9, decay=1.0, norm='inf', loss='CrossEntropy',
                 normalize_grad=False, early_stopping=0, restarts=0, init_noise_generator=None, model=None,
                 save_trajectory=False):
        super().__init__(loss, restarts, num_classes, model=model, save_trajectory=save_trajectory)
        #loss either pass 'CrossEntropy' or 'LogitsDiff' or custom loss function
        self.eps = eps
        self.iterations = iterations
        self.stepsize = stepsize
        self.momentum = momentum
        self.decay = decay
        self.norm = norm
        self.loss = loss
        self.normalize_grad = normalize_grad
        self.early_stopping = early_stopping
        self.init_noise_generator = init_noise_generator

    def __get_trajectory_depth(self):
        return self.iterations + 1

    def get_config_dict(self):
        dict = {}
        dict['type'] = 'ArgminPGD'
        dict['eps'] = self.eps
        dict['iterations'] = self.iterations
        dict['stepsize'] = self.stepsize
        dict['norm'] = self.norm
        if isinstance(self.loss, str):
            dict['loss'] = self.loss
        dict['restarts'] = self.restarts
        #dict['init_sigma'] = self.init_sigma
        return dict

    def perturb_inner(self, x, y, targeted=False):
        l_f = self._get_loss_f(x, y, targeted, 'none')

        best_perts = x.new_empty(x.shape)
        best_losses = 1e13 * x.new_ones(x.shape[0])

        velocity = torch.zeros_like(x)

        #initialize perturbation
        if self.init_noise_generator is None:
            pert = torch.zeros_like(x)
        else:
            pert = self.init_noise_generator(x)

        #trajectory container
        if self.save_trajectory:
            trajectory = torch.zeros((self.iterations + 1,) + x.shape, device=x.device)
            trajectory[0, :] = x
        else:
            trajectory = None

        for i in range(self.iterations + 1):
            pert.requires_grad_(True)
            with torch.enable_grad():
                p_data = x + pert
                out = self.model(p_data)
                loss_expanded = l_f(p_data, out)

                new_best = loss_expanded < best_losses
                best_losses[new_best] = loss_expanded[new_best].clone().detach()
                best_perts[new_best, :] = pert[new_best, :].clone().detach()

                if i == self.iterations:
                    break

                if self.early_stopping > 0:
                    finished, mask = create_early_stopping_mask(out, y, self.early_stopping, targeted)
                    if finished:
                        break
                else:
                    mask = 1.

                loss = torch.mean(loss_expanded)
                grad = torch.autograd.grad(loss, pert)[0]

            with torch.no_grad():
                # pgd on given loss
                if self.normalize_grad:
                    # https://arxiv.org/pdf/1710.06081.pdf the l1 normalization follows the momentum iterative method
                    l1_norm_gradient =  1e-10 + torch.sum(grad.abs().view(x.shape[0], -1), dim=1).view(-1,1,1,1)
                    velocity = self.momentum * velocity + grad / l1_norm_gradient
                    norm_velocity = normalize_perturbation(velocity, self.norm)
                else:
                    # velocity update as in pytorch https://pytorch.org/docs/stable/optim.html
                    velocity = self.momentum * velocity + grad
                    norm_velocity = velocity

                pert = pert - (self.decay**i) * self.stepsize * mask * norm_velocity
                #todo check order
                pert = project_perturbation(pert, self.eps, self.norm)
                pert = torch.clamp(x + pert, 0, 1) - x #box constraint

                if self.save_trajectory:
                    trajectory[i + 1] = x + pert

        final_loss = best_losses
        p_data = (x + best_perts).detach()
        return p_data, final_loss, trajectory


###################################################################################################
def calculate_smart_lr(prev_mean_lr, lr_accepted, lr_decay, iterations, max_lr):
    accepted_idcs = lr_accepted > 0
    if torch.sum(accepted_idcs).item() > 0:
        new_lr = 0.5 * (prev_mean_lr + torch.mean(lr_accepted[lr_accepted > 0]).item())
    else:
        new_lr = prev_mean_lr * ( lr_decay ** iterations )

    new_lr = min(max_lr, new_lr)
    return new_lr


class MonotonePGD(Restart_attack):
    def __init__(self, eps, iterations, stepsize, num_classes, momentum=0.9, lr_smart=False, lr_decay=0.5, lr_gain=1.1,
                 norm='inf', loss='CrossEntropy', normalize_grad=False, early_stopping=0, restarts=0,
                 init_noise_generator=None, model=None, save_trajectory=False):
        super().__init__(loss, restarts, num_classes, model=model, save_trajectory=save_trajectory)
        #loss either pass 'CrossEntropy' or 'LogitsDiff' or custom loss function
        self.eps = eps
        self.iterations = iterations
        self.stepsize = stepsize
        self.momentum = momentum
        self.norm = norm
        self.normalize_grad = normalize_grad
        self.early_stopping = early_stopping
        self.init_noise_generator = init_noise_generator
        self.lr_smart = lr_smart
        self.prev_mean_lr = stepsize
        self.lr_decay = lr_decay #stepsize decay
        self.lr_gain = lr_gain

    def __get_trajectory_depth(self):
        return self.iterations + 1

    def get_config_dict(self):
        dict = {}
        dict['type'] = 'PGD'
        dict['eps'] = self.eps
        dict['iterations'] = self.iterations
        dict['stepsize'] = self.stepsize
        dict['norm'] = self.norm
        if isinstance(self.loss, str):
            dict['loss'] = self.loss
        dict['restarts'] = self.restarts
        #dict['init_sigma'] = self.init_sigma
        dict['lr_gain'] = self.lr_gain
        dict['lr_decay'] = self.lr_decay
        return dict


    def perturb_inner(self, x, y, targeted=False):
        l_f = self._get_loss_f(x, y, targeted, 'none')

        if self.lr_smart:
            lr_accepted = -1 * x.new_ones(self.iterations, x.shape[0])
            lr = self.prev_mean_lr * x.new_ones(x.shape[0])
        else:
            lr = self.stepsize * x.new_ones(x.shape[0])

        #initialize perturbation
        if self.init_noise_generator is None:
            pert = torch.zeros_like(x)
        else:
            pert = self.init_noise_generator(x)
            pert = torch.clamp(x + pert, 0, 1) - x  # box constraint
            pert = project_perturbation(pert, self.eps, self.norm)

        #TODO fix the datatype here !!!
        prev_loss = 1e13 * x.new_ones(x.shape[0], dtype=torch.float)

        prev_pert = pert.clone().detach()
        prev_velocity = torch.zeros_like(pert)
        velocity = torch.zeros_like(pert)

        #trajectory container
        if self.save_trajectory:
            trajectory = torch.zeros((self.iterations + 1,) + x.shape, device=x.device)
            trajectory[0, :] = x
        else:
            trajectory = None

        for i in range(self.iterations + 1):
            pert.requires_grad_(True)
            with torch.enable_grad():
                data = x + pert
                out = self.model(data)

                loss_expanded = l_f(data, out)
                loss = torch.mean(loss_expanded)
                grad = torch.autograd.grad(loss, pert)[0]

            with torch.no_grad():
                loss_increase_idx = loss_expanded > prev_loss

                pert[loss_increase_idx, :] = prev_pert[loss_increase_idx, :].clone().detach()
                loss_expanded[loss_increase_idx] = prev_loss[loss_increase_idx].clone().detach()
                prev_pert = pert.clone().detach()
                prev_loss = loss_expanded
                #previous velocity always holds the last accepted velocity vector
                #velocity the one used for the last update that might have been rejected
                velocity[loss_increase_idx, :] = prev_velocity[loss_increase_idx, :]
                prev_velocity = velocity.clone().detach()

                if i > 0:
                    #use standard lr in firt iteration
                    lr[loss_increase_idx] *= self.lr_decay
                    lr[~loss_increase_idx] *= self.lr_gain

                if i == self.iterations:
                    break

                if self.lr_smart:
                    lr_accepted[i, ~loss_increase_idx] = lr[~loss_increase_idx]

                if self.early_stopping > 0:
                    finished, mask = create_early_stopping_mask(out, y, self.early_stopping, targeted)
                    if finished:
                        break
                else:
                    mask = 1.

                #pgd on given loss
                if self.normalize_grad:
                    if self.momentum > 0:
                        #https://arxiv.org/pdf/1710.06081.pdf the l1 normalization follows the momentum iterative method
                        l1_norm_gradient = 1e-10 + torch.sum(grad.abs().view(x.shape[0], -1), dim=1).view(-1,1,1,1)
                        velocity = self.momentum * velocity + grad / l1_norm_gradient
                    else:
                        velocity = grad
                    norm_velocity = normalize_perturbation(velocity, self.norm)
                else:
                    # velocity update as in pytorch https://pytorch.org/docs/stable/optim.html
                    velocity = self.momentum * velocity + grad
                    norm_velocity = velocity

                pert = pert - mask * lr[:,None,None,None] * norm_velocity
                pert = project_perturbation(pert, self.eps, self.norm)
                pert = torch.clamp(x + pert, 0, 1) - x #box constraint

                if self.save_trajectory:
                    trajectory[i + 1] = x + pert

        if self.lr_smart:
            self.prev_mean_lr = calculate_smart_lr(self.prev_mean_lr, lr_accepted, self.lr_decay, self.iterations, 2 * self.eps)

        return data, loss_expanded, trajectory


###################################################################################################
class MonotoneSteepestDescentPGD(Restart_attack):
    def __init__(self, eps, iterations, stepsize, num_classes, S, low_rank=False,
                 momentum = 0.9, lr_smart = False, lr_decay=0.5, lr_gain=1.1,
                 norm='inf', loss='CrossEntropy', normalize_grad=False, early_stopping=0,
                 restarts=0, init_sigma=None, model=None, save_trajectory=False):
        super().__init__(loss, restarts, num_classes, model=model, save_trajectory=save_trajectory)
        #loss either pass 'CrossEntropy' or 'LogitsDiff' or custom loss function
        self.eps = eps
        self.iterations = iterations
        self.stepsize = stepsize
        self.momentum = momentum
        self.S = S
        self.S_sqrt = S.power(.5)
        self.low_rank = low_rank
        self.norm = norm
        self.loss = loss
        self.normalize_grad = normalize_grad
        self.early_stopping = early_stopping
        self.init_sigma = init_sigma
        self.lr_smart = lr_smart
        self.prev_mean_lr = stepsize
        self.lr_decay = lr_decay #stepsize decay
        self.lr_gain = lr_gain

    def __get_trajectory_depth(self):
        return self.iterations + 1

    def get_config_dict(self):
        dict = {}
        dict['type'] = 'PGD'
        dict['eps'] = self.eps
        dict['iterations'] = self.iterations
        dict['stepsize'] = self.stepsize
        dict['norm'] = self.norm
        if isinstance(self.loss, str):
            dict['loss'] = self.loss
        dict['restarts'] = self.restarts
        dict['init_sigma'] = self.init_sigma
        dict['smart_lr'] = self.lr_smart
        dict['mean_lr'] = self.accu_lr
        dict['lr_gain'] = self.lr_gain
        dict['lr_decay'] = self.lr_decay
        return dict


    def perturb_inner(self, x, y, targeted=False):
        l_f = self._get_loss_f(x, y, targeted, 'none')

        if self.lr_smart:
            lr_accepted = -1 * x.new_ones(self.iterations, x.shape[0])
            lr = self.prev_mean_lr * x.new_ones(x.shape[0])
        else:
            lr = self.stepsize * x.new_ones(x.shape[0])

        prev_loss = 1e13 * x.new_ones(x.shape[0])

        #initialize perturbation
        if self.init_sigma is None:
            pert = torch.zeros_like(x)
        else:
            pert = self.init_sigma * torch.randn_like(x)
            pert = self.S_sqrt(pert.view(x.shape[0], -1), y).view_as(x)

        prev_pert = pert.clone().detach()
        prev_velocity = torch.zeros_like(pert)
        velocity = torch.zeros_like(pert)

        #trajectory container
        if self.save_trajectory:
            trajectory = torch.zeros((self.iterations + 1,) + x.shape, device=x.device)
            trajectory[0, :] = x
        else:
            trajectory = None

        for i in range(self.iterations + 1):
            pert.requires_grad_(True)
            with torch.enable_grad():
                data = x + pert
                out = self.model(data)

                loss_expanded = l_f(data, out)
                loss = torch.mean(loss_expanded)
                grad = torch.autograd.grad(loss, pert)[0]

            with torch.no_grad():
                loss_increase_idx = loss_expanded > prev_loss

                pert[loss_increase_idx, :] = prev_pert[loss_increase_idx, :]
                loss_expanded[loss_increase_idx] = prev_loss[loss_increase_idx]
                prev_pert = pert.clone().detach()
                prev_loss = loss_expanded
                prev_velocity[loss_increase_idx, :] = velocity[loss_increase_idx, :]

                if i > 0:
                    #use standard lr in firt iteration
                    lr[loss_increase_idx] *= self.lr_decay
                    lr[~loss_increase_idx] *= self.lr_gain

                if i == self.iterations:
                    break

                if self.lr_smart:
                    lr_accepted[i, ~loss_increase_idx] = lr[~loss_increase_idx]

                if self.early_stopping > 0:
                    conf, pred = torch.max(torch.nn.functional.softmax(out, dim=1), 1)
                    conf_mask = conf > self.early_stopping
                    if targeted:
                        correct_mask = torch.eq(y, pred)
                    else:
                        correct_mask = (~torch.eq(y, pred))

                    mask = 1. - (conf_mask & correct_mask).float()
                    mask = mask * lr

                    if sum(1.0 - mask) == x.shape[0]:
                        break
                else:
                    mask = lr

                mask = mask[(..., ) + (None, ) * 3]

                # steepest descent direction is S * grad
                if self.low_rank:
                    grad = (self.S(grad.view(x.shape[0], -1), y)).view_as(grad)
                else:
                    grad = torch.bmm(self.S[y, :], grad.view(x.shape[0], -1, 1)).view_as(grad)
                    
                #pgd on given loss
                if self.normalize_grad:
                    #https://arxiv.org/pdf/1710.06081.pdf the l1 normalization follows the momentum iterative method
                    l1_norm_gradient = grad / torch.sum( grad.abs().view(x.shape[0], -1) )
                    velocity = self.momentum * prev_velocity + l1_norm_gradient
                    norm_velocity = normalize_perturbation(velocity, self.norm)
                else:
                    # velocity update as in pytorch https://pytorch.org/docs/stable/optim.html
                    velocity = self.momentum * prev_velocity + grad
                    norm_velocity = velocity

                pert = pert - mask * norm_velocity
                pert = torch.clamp(x + pert, 0, 1) - x #box constraint
                pert = project_perturbation(pert, self.eps, self.norm)

                if self.save_trajectory:
                    trajectory[i + 1] = data

        if self.lr_smart:
            self.prev_mean_lr = calculate_smart_lr(self.prev_mean_lr, lr_accepted, self.lr_decay, self.iterations, 2 * self.eps)

        return data, loss, trajectory
