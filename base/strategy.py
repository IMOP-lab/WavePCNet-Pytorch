import importlib
from torch import nn
import math

from torch.nn import functional as F
from torch.optim import *
import torch.optim.lr_scheduler as sche

sgd_base_config = {
    'optim': 'SGD',
    'lr': 5e-3,
    'agg_batch': 32,
    'epoch': 40,
    }
def sgd_base(optimizer, current_iter, total_iter, config):
    if (current_iter / total_iter) < 0.5:
        factor = 1
    elif (current_iter / total_iter) < 0.75:
        factor = 0.1
    else:
        factor = 0.01
    optimizer.param_groups[0]['lr'] = factor * config['lr'] * 0.1
    optimizer.param_groups[1]['lr'] = factor * config['lr']

sgd_poly_config = {
    'optim': 'SGD',
    'lr': 1e-3,
    'agg_batch': 32,
    'epoch': 40,
    }
def sgd_poly(optimizer, current_iter, total_iter, config):
    factor = pow(1 - (current_iter / total_iter), 0.9)
        
    optimizer.param_groups[0]['lr'] = factor * config['lr'] * 0.1
    optimizer.param_groups[1]['lr'] = factor * config['lr']

adam_base_config = {
    'optim': 'Adam',
    'lr': 1e-3,
    'agg_batch': 32,
    'epoch': 40,
}
def adam_base(optimizer, current_iter, total_iter, config):
    if (current_iter / total_iter) < 0.2:
        factor = 1
    elif (current_iter / total_iter) < 0.5:
        factor = 0.1
    else:
        factor = 0.01
        
    optimizer.param_groups[0]['lr'] = factor * config['lr'] * 0.1
    optimizer.param_groups[1]['lr'] = factor * config['lr']

def Strategy(model, config):
    strategy = config['strategy']
    stra_config = eval(strategy + '_config')
    config.update(stra_config)
    
    if 'params' in config.keys():
        module_lr = [{'params' : getattr(model, p[0]).parameters(), 'lr' : p[1]} for p in config['params']]
    else:
        encoder = []
        others = []
        for param in model.named_parameters():
            if 'encoder.' in param[0]:
                encoder.append(param[1])
            else:
                others.append(param[1])
        if len(encoder) == 0:
            print("Warning: parameters in encoder not found!")
        module_lr = [{'params' : encoder, 'lr' : config['lr']*0.1}, {'params' : others, 'lr' : config['lr']}]
        
    optim = config['optim']
    if optim == 'SGD':
        optimizer = SGD(params=module_lr, lr=config['lr'], momentum=0.9, weight_decay=0.0005)
    elif optim == 'Adam':
        optimizer = Adam(params=module_lr, lr=config['lr'], weight_decay=0.0005)
    elif optim == 'AdamW':
        optimizer = AdamW(params=module_lr, lr = config['lr'], weight_decay=0.05)
        
    schedule = eval(strategy)
    return optimizer, schedule
