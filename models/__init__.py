import torch.nn as nn
from catalyst.contrib.models.segmentation import Unet

from .unet_resnet import UNetResNet
from .linknet import LinkNet34


model_map = {
    'Unet': Unet,
    'UNetResNet': UNetResNet,
    'LinkNet34': LinkNet34
}


def get_model(name: str, **kwargs) -> nn.Module:
    return model_map[name](**kwargs)


