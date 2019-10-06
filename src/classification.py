import os
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import catalyst.dl.callbacks as cbks
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import CheckpointCallback
from catalyst.utils.seed import set_global_seed

from .utils import (load_config, optimizers_map, get_optimizer,
                    get_loss, get_dataset)
from .models import get_model
from .metrics import AccuracyCallback


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', '-c',
    dest='config',
    type=str,
    help='config file'
)
parser.add_argument(
    '--device',
    dest='device',
    type=int,
    help='device to use (<0 - cpu, other (>0) - gpu numbers)',
    default=0
)
parser.add_argument(
    '--logs', '-l',
    dest='logdir',
    type=str,
    help='directory to store logs'
)
parser.add_argument(
    '--checkpoint', '-s',
    dest='checkpoint',
    type=str,
    help='model state to use for training',
    default=''
)
args = vars(parser.parse_args())

CONFIG_FILE = args['config']
DEVICE = torch.device('cuda:{}'.format(args['device']) if args['device'] >= 0 else 'cpu')
LOGDIR = args['logdir']
CHECKPOINT = args['checkpoint']


def main() -> None:
    config = load_config(CONFIG_FILE)
    train_config = config['train']
    
    num_epochs = config.get('num epochs', 2)
    random_state = config.get('random state', 2019)
    num_workers = config.get('num workers', 6)
    batch_size = config['batch size']
    
    train_dataset = get_dataset(**config['train'])
    valid_dataset = get_dataset(**config['validation'])
    
    data_loaders = OrderedDict()
    data_loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    data_loaders['valid'] = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    set_global_seed(random_state)
    
    model = get_model(**config['model'])
    
    if CHECKPOINT != '' and os.path.exists(CHECKPOINT):
        checkpoint_state = torch.load(CHECKPOINT)['model_state_dict']
        model.load_state_dict(checkpoint_state)
        print(f'Using {CHECKPOINT} checkpoint', flush=True)
    
    model = model.to(DEVICE)
    
    model_optimizer = get_optimizer(model.parameters(), **config['optimizer'])
    
    loss_function = get_loss(**config['loss'])
    metric = config.get('metric', 'loss')
    is_metric_minimization = config.get('minimize metric', True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        model_optimizer, 
        mode='min' if is_metric_minimization else 'max',
        patience=3,
        factor=0.2, 
        verbose=True
    )

    runner = SupervisedRunner(device=DEVICE)
    runner.train(
        model=model,
        criterion=loss_function,
        optimizer=model_optimizer,
        loaders=data_loaders,
        logdir=LOGDIR,
        callbacks=[
            cbks.AUCCallback(),
            cbks.F1ScoreCallback(),
            AccuracyCallback(),
            cbks.CriterionCallback(),
            CheckpointCallback(save_n_best=3)
        ],
        scheduler=scheduler,
        verbose=True,
        minimize_metric=is_metric_minimization,
        num_epochs=num_epochs,
        main_metric=metric
    )


if __name__ == '__main__':
    main()
