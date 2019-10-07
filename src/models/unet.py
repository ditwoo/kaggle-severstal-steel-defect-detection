import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck


class ResNetEncoder(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        del self.fc

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1, x0]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        super().load_state_dict(state_dict, **kwargs)


encoder_architectures = {
    "resnet18": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
                "input_space": "RGB",
                "input_size": (3, 224, 224),
                "input_range": [0, 1],
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
                "num_classes": 1000,
            }
        },
        "out_shapes": (512, 256, 128, 64, 64),
        "params": {"block": BasicBlock, "layers": [2, 2, 2, 2]},
    },
    "resnet34": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
                "input_space": "RGB",
                "input_size": (3, 224, 224),
                "input_range": [0, 1],
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
                "num_classes": 1000,
            }
        },
        "out_shapes": (512, 256, 128, 64, 64),
        "params": {"block": BasicBlock, "layers": [3, 4, 6, 3]},
    },
    "resnet50": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
                "input_space": "RGB",
                "input_size": (3, 224, 224),
                "input_range": [0, 1],
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
                "num_classes": 1000,
            }
        },
        "out_shapes": (2048, 1024, 512, 256, 64),
        "params": {"block": Bottleneck, "layers": [3, 4, 6, 3]},
    },
    "resnet101": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
                "input_space": "RGB",
                "input_size": (3, 224, 224),
                "input_range": [0, 1],
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
                "num_classes": 1000,
            }
        },
        "out_shapes": (2048, 1024, 512, 256, 64),
        "params": {"block": Bottleneck, "layers": [3, 4, 23, 3]},
    },
    "resnet152": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
                "input_space": "RGB",
                "input_size": (3, 224, 224),
                "input_range": [0, 1],
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
                "num_classes": 1000,
            }
        },
        "out_shapes": (2048, 1024, 512, 256, 64),
        "params": {"block": Bottleneck, "layers": [3, 8, 36, 3]},
    },
}


def get_encoder(name: str, encoder_weights: str = None) -> ResNetEncoder:
    Encoder = encoder_architectures[name]["encoder"]
    encoder = Encoder(**encoder_architectures[name]["params"])
    encoder.out_shapes = encoder_architectures[name]["out_shapes"]

    if encoder_weights is not None:
        settings = encoder_architectures[name]["pretrained_settings"][encoder_weights]
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    return encoder


class Conv2dReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
        **batchnorm_params
    ):

        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=not (use_batchnorm),
            ),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            Conv2dReLU(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            ),
            Conv2dReLU(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            ),
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class CenterBlock(DecoderBlock):
    def forward(self, x):
        return self.block(x)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels=(256, 128, 64, 32, 16),
        num_classes: int = 1,
        use_batchnorm: bool = False,
        center: bool = False,
    ):
        super(UnetDecoder, self).__init__()

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        in_channels = self.__compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(
            in_channels[0], out_channels[0], use_batchnorm=use_batchnorm
        )
        self.layer2 = DecoderBlock(
            in_channels[1], out_channels[1], use_batchnorm=use_batchnorm
        )
        self.layer3 = DecoderBlock(
            in_channels[2], out_channels[2], use_batchnorm=use_batchnorm
        )
        self.layer4 = DecoderBlock(
            in_channels[3], out_channels[3], use_batchnorm=use_batchnorm
        )
        self.layer5 = DecoderBlock(
            in_channels[4], out_channels[4], use_batchnorm=use_batchnorm
        )
        self.final_conv = nn.Conv2d(out_channels[4], num_classes, kernel_size=(1, 1))

        self.__initialize()

    def __initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __compute_channels(self, encoder_channels, decoder_channels):
        channels = (
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        )
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        return x


class ResUnet(nn.Module):
    def __init__(
        self,
        encoder: str,
        use_batchnorm: bool = True,
        decoder_channels=(256, 128, 64, 32, 16),
        num_classes: int = 1,
        activation: str = "sigmoid",
        center: bool = False,
    ):
        super(ResUnet, self).__init__()

        self.encoder = get_encoder(encoder, encoder_weights="imagenet")
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_shapes,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            use_batchnorm=use_batchnorm,
            center=center,
        )

        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`
        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)
        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)

        return x
