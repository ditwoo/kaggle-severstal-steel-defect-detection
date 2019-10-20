import torch
import torch.nn as nn
from torchvision import models
from typing import List


_resnet_backbones = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=False):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels)
            )

    def forward(self, x):
        return self.block(torch.cat(x, 1))


class DSVBlock(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(DSVBlock, self).__init__()
        self.dsv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

    def forward(self, input):
        return self.dsv(input)


class SCseUnet(nn.Module):
    def __init__(self, 
                 backbone: str = 'resnet18', 
                 num_classes: int = 1, 
                 is_deconv: bool = True,
                 pretrained: bool = False):
        self.inplanes = 64
        super().__init__()

        encoder = _resnet_backbones[backbone](pretrained=pretrained)
        filters = [
            encoder.layer1[-1].conv2.out_channels,
            encoder.layer2[-1].conv2.out_channels,
            encoder.layer3[-1].conv2.out_channels,
            encoder.layer4[-1].conv2.out_channels,
        ]
        dfilters = filters[:3]
        self.pool = nn.MaxPool2d(2)
        self.preprocess = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64))
        self.layer1 = encoder.layer1  # -> 256
        self.layer2 = encoder.layer2  # -> 512
        self.layer3 = encoder.layer3  # -> 1024
        self.layer4 = encoder.layer4  # -> 2048

        center_out = filters[2] // 2
        self.center = nn.Sequential(
            ConvRelu(filters[3], center_out),
            nn.ConvTranspose2d(center_out, center_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            DecoderBlock(filters[3] + filters[2] // 2, dfilters[2], dfilters[2], is_deconv=is_deconv),
            SCse(dfilters[2]),
        )
        self.dec3 = nn.Sequential(
            DecoderBlock(filters[2] + dfilters[2], dfilters[1], dfilters[1], is_deconv=is_deconv),
            SCse(dfilters[1]),
        )
        self.dec2 = nn.Sequential(
            DecoderBlock(filters[1] + dfilters[1], dfilters[0], dfilters[0], is_deconv=is_deconv),
            SCse(dfilters[0]),
        )
        self.dec1 = nn.Sequential(
            DecoderBlock(filters[0] + dfilters[0], filters[0], filters[0], is_deconv=is_deconv),
            SCse(dfilters[0]),
        )
        self.final = nn.Sequential(
            ConvRelu(filters[0] + 64, 64),
            nn.Conv2d(64, num_classes, 1)
        )

        # self.dsv4 = DSVBlock(dfilters[2], num_classes, 8)
        # self.dsv3 = DSVBlock(dfilters[1], num_classes, 4)
        # self.dsv2 = DSVBlock(dfilters[0], num_classes, 2)
        # self.dsv1 = DSVBlock(filters[0], num_classes, 1)
        # self.dsv_out = nn.Conv2d(num_classes * 4, num_classes, 1)

    def forward(self, x):
        preprocessed = self.preprocess(x)
        conv1 = self.layer1(self.pool(preprocessed))
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        conv4 = self.layer4(conv3)

        center = self.center(conv4)

        dec4 = self.dec4([center, conv4])
        dec3 = self.dec3([dec4, conv3])
        dec2 = self.dec2([dec3, conv2])
        dec1 = self.dec1([dec2, conv1])
        out = self.final(torch.cat([dec1, preprocessed], 1))

        return out

        # dsv4 = self.dsv4(dec4)
        # dsv3 = self.dsv3(dec3)
        # dsv2 = self.dsv2(dec2)
        # dsv1 = self.dsv1(dec1)
        # dsv_out = self.dsv_out(torch.cat([dsv1,dsv2,dsv3,dsv4], 1))

        # return {'mask': out, 'dsv': dsv_out}


if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 1600)
    m = SCseUnet('resnet34', num_classes=4)
    m.eval()
    with torch.no_grad():
        out = m(x)
    print('input: ', x.shape)
    print('out:', out.shape)
    # print('mask:', out['mask'].shape)
    # print('dsv:', out['dsv'].shape)

