import json
import yaml

import numpy as np
from pandas import read_csv

from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.nn.modules.loss import _Loss

import albumentations.pytorch
from albumentations.core.serialization import from_dict

from dataset import SteelDataset
from losses import (BinaryDiceLoss, BinaryDiceLogLoss,
                    MulticlassDiceLoss, CCE,
                    WeightedLoss, JointLoss, TverskyLoss)

optimizers_map = {
    'Adam': optim.Adam,
    'Adamax': optim.Adamax,
    'ASGD': optim.ASGD,
    'RMSprop': optim.RMSprop,
    'SGD': optim.SGD
}

losses_map = {
    'CrossEntropyLoss': CrossEntropyLoss,
    'BinaryDiceLoss': BinaryDiceLoss,
    'BinaryDiceLogLoss': BinaryDiceLogLoss,
    'MulticlassDiceLoss': MulticlassDiceLoss,
    'CCE': CCE,
    'WeightedLoss': WeightedLoss,
    'JointLoss': JointLoss,
    'TverskyLoss': TverskyLoss
}


def _rle_str2arr(rle_str: str) -> np.ndarray:
    res = []
    if not (rle_str != rle_str):
        res = list(map(int, rle_str.split(' ')))
    return np.array(res)


def load_transforms(transforms: dict):
    return from_dict(transforms)


def get_dataset(file: str, transforms: dict) -> SteelDataset:
    classes = [f'cls{i}' for i in range(1, 5)]
    
    df = read_csv(file)
    for _cls in classes:
        df[_cls] = df[_cls].apply(_rle_str2arr)
    
    data_transforms = load_transforms(transforms)
    
    dataset = SteelDataset(
        images=df['Image'].values,
        rles=df[classes].values,
        transforms=data_transforms
    )
    return dataset
    
    
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
    