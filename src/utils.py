import json
import yaml

import numpy as np
from pandas import read_csv

from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
import torch.optim as optim
from torch.nn.modules.loss import _Loss

from optimizers import RAdam, PlainRAdam, AdamW
from datasets import get_dataset
from losses import (BinaryDiceLoss, BinaryDiceLogLoss,
                    MulticlassDiceLoss, CCE,
                    WeightedLoss, JointLoss, TverskyLoss)



optimizers_map = {
    'Adam': optim.Adam,
    'RAdam': RAdam,
    'PlainRAdam': PlainRAdam,
    'AdamW': AdamW,
    'Adamax': optim.Adamax,
    'ASGD': optim.ASGD,
    'RMSprop': optim.RMSprop,
    'SGD': optim.SGD
}

losses_map = {
    'BCELoss': BCELoss,
    'BCEWithLogitsLoss': BCEWithLogitsLoss,
    'CrossEntropyLoss': CrossEntropyLoss,
    'BinaryDiceLoss': BinaryDiceLoss,
    'BinaryDiceLogLoss': BinaryDiceLogLoss,
    'MulticlassDiceLoss': MulticlassDiceLoss,
    'CCE': CCE,
    'WeightedLoss': WeightedLoss,
    'JointLoss': JointLoss,
    'TverskyLoss': TverskyLoss
}
    
    
def load_config(filename: str) -> dict:
    if filename.endswith('yaml') or filename.endswith('yml'):
        reader = yaml.safe_load
    else:
        reader = json.load
        
    with open(filename, 'r') as file_content:
        config = reader(file_content)
    
    return config


def get_optimizer(model_params, name: str, **kwargs) -> optim.Optimizer:
    optimizer = optimizers_map.get(name)
    
    if optimizer:
        optimizer = optimizer(model_params, **kwargs)
    else:
        optimizer = optim.Adam(model_params)

    return optimizer
    
    
def get_loss(name: str, **kwargs) -> _Loss:
    if name in {'JointLoss'}:
        kwargs['first'] = get_loss(**kwargs['first'])
        kwargs['second'] = get_loss(**kwargs['second'])
        return JointLoss(**kwargs)
    return losses_map.get(name)(**kwargs)
    