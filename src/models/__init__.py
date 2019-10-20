from .unet import ResUnet, ResUnetScSeDecoded
from .unet_resnet import UNetResNet
from .linknet import LinkNet34
from .detection import DenseNetDetector, ResnetDetector
from .scse_unet import SCseUnet
from .qunet import QUnet
from .efficientnet import EfficientUnet
from torchvision.models import resnet34
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

__all__ = (
    "ResUnet", "UNetResNet", "LinkNet34", "DenseNetDetector",
    "ResnetDetector", "resnet34", "SCseUnet", "ResUnetScSeDecoded",
    "QUnet", "ModelFromCheckpoint", "EfficientUnet"
)
