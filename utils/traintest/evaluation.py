import numpy as np
np.seterr(all='ignore')

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torchvision

import scipy.special
import matplotlib.pyplot as plt
import matplotlib
import sklearn.metrics
import datetime
import os


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def auroc_conservative(values_in, values_out):
    s = 0
    if type(values_in)==torch.Tensor:
        values_in = values_in.numpy()
    if type(values_out)==torch.Tensor:
        values_out = values_out.numpy()

    for i in range(len(values_in)):
        s += (values_out < values_in[i]).sum() #+ 0.5*(values_out == values_in[i]).sum()
    s = s.astype(float)
    s /= float( len(values_in)*len(values_out) )
    return s 


def auroc(values_in, values_out):
    y_true = len(values_in)*[1] + len(values_out)*[0]
    y_score = np.concatenate([np.nan_to_num(values_in, nan=0.0), np.nan_to_num(values_out, nan=0.0)])
    return sklearn.metrics.roc_auc_score(y_true, y_score)


# def auroc_memory(values_in, values_out):
#     s = 0
#     for i in range(len(values_in)):
#         s += (values_out < values_in[i]).sum() + 0.5*(values_out == values_in[i]).sum()
#     s /= len(values_in)*len(values_out)
#     return s


def fpr_at_tpr(values_in, values_out, tpr):
    if type(values_in)==torch.Tensor:
        values_in = values_in.numpy()
    if type(values_out)==torch.Tensor:
        values_out = values_out.numpy()
    
    t = np.quantile(values_in, (1-tpr))
    fpr = (values_out >= t).mean()
    return fpr
    


def auprc(values_in, values_out):
    y_true = len(values_in)*[1] + len(values_out)*[0]
    y_score = np.concatenate([values_in, values_out])
    return sklearn.metrics.average_precision_score(y_true, y_score)


def accuracy(P, L):
    """Mean euclidean distance between two N✕2 numpy arrays"""
    C = (P == L)
    corr = np.sum(C)
    acc = corr/len(C)
    return acc


def get_accuracy(model, device, loader, from_logits=True):
    conf = []
    acc = 0
    num = 0
    reduction = ( lambda x: torch.log_softmax(x, dim=1) ) if from_logits else ( lambda x: x )
    for x, y in loader:
        pred = reduction(model(x.to(device))).detach().cpu()
        c, pr_cls = pred.max(1)
        conf.append(c.exp())
        acc += (pr_cls==y).sum().item()
        num += pr_cls.shape[0]
    return acc / num, torch.cat(conf, dim=0)


def get_conf(model, device, loader, from_logits=True):
    conf = []
    num = 0
    reduction = ( lambda x: torch.log_softmax(x, dim=1) ) if from_logits else ( lambda x: x )
    for x, y in loader:
        pred = reduction(model(x.to(device))).detach().cpu()
        c, pr_cls = pred.max(1)
        conf.append(c.exp())
    return torch.cat(conf, dim=0)


def get_confusion_matrix(model, device, loader, from_logits=True):
    classes = len(loader.dataset.classes)
    confusion_matrix = torch.zeros(classes, classes)
    confidence_matrix = torch.zeros(classes, classes)
    
    reduction = ( lambda x: torch.softmax(x, dim=1) ) if from_logits else (lambda x: x.exp())
    with torch.no_grad():
        for x, y in loader:
            outputs = reduction(model(x.to(device))).detach().cpu()
            conf, pred = outputs.max(1)
            for t, p, c in zip(y, pred, conf):
                confusion_matrix[t, p.long()] += 1
                confidence_matrix[t, p.long()] += c
                
    confidence_matrix = confidence_matrix / confusion_matrix
    index = torch.isnan(confidence_matrix)
    confidence_matrix[index] = 0.0
    return confusion_matrix, confidence_matrix


def get_class_wise_accuracy(model, device, loader):
    M, _ = get_confusion_matrix(model, device, loader)
    return M.diag()/M.sum(1)


def get_output(model, device, loader, max_batches=20000):
    out = []
    with torch.no_grad():
        for idx, (x, _) in enumerate(loader):
            out.append(model(x.to(device)).detach().cpu())
            if idx>=max_batches:
                break
    return torch.cat(out, dim=0)


def get_output_ub(model, device, loader, eps=0.01):
    out = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            l, u = torch.clamp(x-eps, 0, 1), torch.clamp(x+eps, 0, 1)
            out.append(model.ibp_forward(l, u)[1].detach().cpu())
    return torch.cat(out, dim=0)
    

def get_log_conf(output):
    pred = torch.log_softmax(output, dim=1)
    log_conf = pred.max(1)[0]
    return log_conf


def get_neg_energy(output):
    neg_energy = torch.logsumexp(output, dim=1)
    return neg_energy


def logsumexp(x, axis=-1):
    return scipy.special.logsumexp(x, axis=axis)


def get_ub_log_conf(model, device, loader, eps, num_classes=10):
    ud_logit_out = []
    for x, _ in loader:
        x = x.to(device)
        _, _, ud_logit_out_batch = model.ibp_elision_forward(torch.clamp(x - eps, 0, 1), torch.clamp(x + eps, 0, 1), num_classes) 
        ud_logit_out_batch = ud_logit_out_batch.detach().cpu().numpy()
        ud_logit_out.append(ud_logit_out_batch)
        
    ud_logit_out = np.concatenate(ud_logit_out, axis=0)
    
    ub_el_out_log_confidences = ub_log_confs_from_ud_logits(ud_logit_out, force_diag_0=False)
    return np.nan_to_num(ub_el_out_log_confidences)


def get_ub_neg_energy(model, device, loader, eps, num_classes=10):
    ub_logit = []
    for x, _ in loader:
        x = x.to(device)
        lb, ub = model.ibp_forward(torch.clamp(x - eps, 0, 1), torch.clamp(x + eps, 0, 1)) 
        ub_out_batch = ub.detach().cpu().max(1)[0].numpy()
        ub_logit.append(ub_out_batch)
        
    ub_logit = np.concatenate(ub_logit, axis=0) + np.log(float(num_classes))
    
    return np.nan_to_num(ub_logit)
    

def log_confs_from_logits(logits):
    logits_normalized = logits - logits.max(axis=-1, keepdims=True)
    log_confidences = -logsumexp(logits_normalized, axis=-1)
    return log_confidences

def right_and_wrong_confidences_from_logits(logits, labels):
    #logits_normalized_by_label = logits - logits[:,labels]
    probabilities = softmax(logits, axis=-1)  
    right_confidences = np.copy(probabilities[range(probabilities.shape[0]), labels])
    probabilities[range(probabilities.shape[0]), labels] = 0
    wrong_confidences = probabilities.max(axis=-1)
    return right_confidences, wrong_confidences

def ub_log_confs_from_ud_logits(ud_logits, force_diag_0=False): #upper bound differences matrix
    if force_diag_0: #with elision, this is already given
        for i in range(ud_logits.shape[-1]): 
            ud_logits[:, i, i] = 0
    ub_log_probs = -logsumexp(-ud_logits, axis=-1)
    ub_log_confs = np.amax(ub_log_probs, axis=-1)
    return ub_log_confs
    
def conf_stats_from_log_confs(log_confs, th, k):
    confidences = np.exp(np.nan_to_num(log_confs))
    confidence_mean = np.mean(confidences)
    confidence_median = np.median(confidences)
    confidences_below_th = np.sum(confidences < th)
    confidences_above_th = np.sum(confidences > th)
    lowest_conf_indices = confidences.argsort()[:k]
    highest_conf_indices = (-confidences).argsort()[:k]
    return confidences, confidence_mean, confidence_median, confidences_below_th, confidences_above_th, lowest_conf_indices, highest_conf_indices 
        
def accuracy_above(logits, labels, th):
    log_confs = log_confs_from_logits(logits)
    confidences = np.exp(np.nan_to_num(log_confs))
    above = confidences > th
    pred_classes = np.argmax(logits, axis=1)
    if np.sum(above) == 0:
        return None
    acc_above = accuracy(pred_classes[above], labels[above])
    return acc_above

def frac_above(logits, labels, th):
    log_confs = log_confs_from_logits(logits)
    confidences = np.exp(np.nan_to_num(log_confs))
    above = confidences > th
    f_above = sum(above) / len(logits)
    return f_above
    