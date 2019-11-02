from .unet import ResUnet, ResUnetScSeDecoded
from .unet_resnet import UNetResNet
from .linknet import LinkNet34
from .detection import DenseNetDetector, ResnetDetector
from .scse_unet import SCseUnet
from .qunet import QUnet
from .efficientnet import EfficientUnet
from torchvision.models import resnet34
from torchvision.models import resnet as rn
from torchvision.models import densenet as dn
import yaml
import torch
from catalyst.dl import registry


# class ModelFromCheckpoint(Module):
#     def __init__(self, config: str, checkpoint: str):
#         with open(config, "r") as f:
#             config = yaml.load(f, Loader=yaml.FullLoader)
        
#         self.model: Module = registry.MODELS.get_from_params(**config["model_params"])
#         checkpoint = torch.load(checkpoint)
#         self.model.load_state_dict(checkpoint["model_state_dict"])

#     def forward(self, *args, **kwargs):
#         return self.model.forward(*args, **kwargs)


def ModelFromCheckpoint(config: str, checkpoint: str) -> torch.nn.Module:
    with open(config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model: torch.nn.Module = registry.MODELS.get_from_params(**config["model_params"])
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def PretrainedResnet(arch: str, num_classes: int) -> torch.nn.Module:
    models = {
        'resnet18': rn.resnet18,
        'resnet34': rn.resnet34,
        'resnet50': rn.resnet50, 
        'resnet101': rn.resnet101,
        'resnet152': rn.resnet152, 
        # 'resnext50_32x4d': rn.resnext50_32x4d, 
        # 'resnext101_32x8d': rn.resnext101_32x8d,
        # 'wide_resnet50_2': rn.wide_resnet50_2, 
        # 'wide_resnet101_2': rn.wide_resnet101_2,
    }
    if arch not in models:
        raise KeyError(f"Unknown model architectures - {arch}")

    m: rn.ResNet = models[arch](pretrained=True)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m


def PretrainedDensenet(arch: str, num_classes: int) -> torch.nn.Module:
    models = {
        'densenet121': dn.densenet121,
        'densenet169': dn.densenet169,
        'densenet201': dn.densenet201,
        'densenet161': dn.densenet161,
    }
    if arch not in models:
        raise KeyError(f"Unknown model architectures - {arch}")

    m: dn.DenseNet = models[arch](pretrained=True)
    m.classifier = torch.nn.Linear(m.classifier.in_features, num_classes)
    return m


__all__ = (
    "ResUnet", "UNetResNet", "LinkNet34", "DenseNetDetector",
    "ResnetDetector", "resnet34", "SCseUnet", "ResUnetScSeDecoded",
    "QUnet", "ModelFromCheckpoint", "EfficientUnet", "PretrainedResnet"
)
