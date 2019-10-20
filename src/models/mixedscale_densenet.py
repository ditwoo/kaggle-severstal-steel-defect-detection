import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/ansleliu/LightNet/blob/master/models/mixscaledensenet.py


class InPlaceABN(nn.Module):
    """InPlace Activated Batch Normalization"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu", slope=0.01):
        """Creates an InPlace Activated Batch Normalization module
        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(InPlaceABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)


class ASPPInPlaceABNBlock(nn.Module):
    def __init__(self, in_chs, out_chs, feat_res=(56, 112),
                 up_ratio=2, aspp_sec=(12, 24, 36), norm_act=ABN):
        super(ASPPInPlaceABNBlock, self).__init__()

        self.in_norm = norm_act(in_chs)
        self.gave_pool = nn.Sequential(OrderedDict([("gavg", nn.AdaptiveAvgPool2d((1, 1))),
                                                    ("conv1_0", nn.Conv2d(in_chs, out_chs,
                                                                          kernel_size=1, stride=1, padding=0,
                                                                          groups=1, bias=False, dilation=1)),
                                                    ("up0", nn.Upsample(size=feat_res, mode='bilinear'))]))

        self.conv1x1 = nn.Sequential(OrderedDict([("conv1_1", nn.Conv2d(in_chs, out_chs, kernel_size=1,
                                                                        stride=1, padding=0, bias=False,
                                                                        groups=1, dilation=1))]))

        self.aspp_bra1 = nn.Sequential(OrderedDict([("conv2_1", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_sec[0], bias=False,
                                                                          groups=1, dilation=aspp_sec[0]))]))

        self.aspp_bra2 = nn.Sequential(OrderedDict([("conv2_2", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_sec[1], bias=False,
                                                                          groups=1, dilation=aspp_sec[1]))]))

        self.aspp_bra3 = nn.Sequential(OrderedDict([("conv2_3", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_sec[2], bias=False,
                                                                          groups=1, dilation=aspp_sec[2]))]))

        self.aspp_catdown = nn.Sequential(OrderedDict([("norm_act", norm_act(5*out_chs)),
                                                       ("conv_down", nn.Conv2d(5*out_chs, out_chs, kernel_size=1,
                                                                               stride=1, padding=1, bias=False,
                                                                               groups=1, dilation=1)),
                                                       ("dropout", nn.Dropout2d(p=0.2, inplace=True))]))

        self.upsampling = nn.Upsample(size=(int(feat_res[0]*up_ratio), int(feat_res[1]*up_ratio)), mode='bilinear')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # channel_shuffle: shuffle channels in groups
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    @staticmethod
    def _channel_shuffle(x, groups):
        """
        Channel shuffle operation
        :param x: input tensor
        :param groups: split channels into groups
        :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()

        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batch_size, groups, channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)

        return x

    def forward(self, x):
        x = self.in_norm(x)
        x = torch.cat([self.gave_pool(x),
                       self.conv1x1(x),
                       self.aspp_bra1(x),
                       self.aspp_bra2(x),
                       self.aspp_bra3(x)], dim=1)

        out = self.aspp_catdown(x)
        return out, self.upsampling(out)


class InPlaceABN(nn.Module):
    """InPlace Activated Batch Normalization"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu", slope=0.01):
        """Creates an InPlace Activated Batch Normalization module
        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(InPlaceABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, x):
        return inplace_abn(x, self.weight, self.bias, autograd.Variable(self.running_mean),
                           autograd.Variable(self.running_var), self.training, self.momentum, self.eps,
                           self.activation, self.slope)

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
              ' affine={affine}, activation={activation}'
        if self.activation == "leaky_relu":
            rep += ' slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)
    
    
class InPlaceABNWrapper(nn.Module):
    """Wrapper module to make `InPlaceABN` compatible with `ABN`"""

    def __init__(self, *args, **kwargs):
        super(InPlaceABNWrapper, self).__init__()
        self.bn = InPlaceABN(*args, **kwargs)

    def forward(self, input):
        return self.bn(input)

    
class DenseModule(nn.Module):
    def __init__(self, in_chns, squeeze_ratio, out_chns, n_layers, dilate_sec=(1, 2, 4, 8, 16), norm_act=ABN):
        super(DenseModule, self).__init__()
        self.n_layers = n_layers
        self.mid_out = int(in_chns * squeeze_ratio)

        self.convs1 = nn.ModuleList()
        self.convs3 = nn.ModuleList()

        for idx in range(self.n_layers):
            dilate = dilate_sec[idx % len(dilate_sec)]
            self.last_channel = in_chns + idx * out_chns

            """
            self.convs1.append(nn.Sequential(OrderedDict([
                ("bn", norm_act(self.last_channel)),
                ("conv", nn.Conv2d(self.last_channel, self.mid_out, 1, bias=False))
            ])))
            """

            self.convs3.append(nn.Sequential(OrderedDict([
                ("bn", norm_act(self.last_channel)),
                ("conv", nn.Conv2d(self.last_channel, out_chns, kernel_size=3, stride=1,
                                   padding=dilate, dilation=dilate, bias=False))
            ])))

    @property
    def out_channels(self):
        return self.last_channel + 1

    def forward(self, x):
        inputs = [x]
        for i in range(self.n_layers):
            x = torch.cat(inputs, dim=1)
            # x = self.convs1[i](x)
            x = self.convs3[i](x)
            inputs += [x]

        return torch.cat(inputs, dim=1)

    
class MixedScaleDenseNet(nn.Module):
    """
    Mixed Scale Dense Network
    """
    def __init__(self, n_class=19, in_size=(448, 896), num_layers=128, in_chns=32, squeeze_ratio=1.0/32, out_chns=1,
                 dilate_sec=(1, 2, 4, 8, 4, 2), aspp_sec=(24, 48, 72), norm_act=InPlaceABN):
        """
        MixedScaleDenseNet: Mixed Scale Dense Network
        :param n_class:    (int) Number of classes
        :param in_size:    (tuple or int) Size of the input image feed to the network
        :param num_layers: (int) Number of layers used in the mixed scale dense block/stage
        :param in_chns:    (int) Input channels of the mixed scale dense block/stage
        :param out_chns:   (int) Output channels of each Conv used in the mixed scale dense block/stage
        :param dilate_sec: (tuple) Dilation rates used in the mixed scale dense block/stage
        :param aspp_sec:   (tuple) Dilation rates used in ASPP
        :param norm_act:   (object) Batch Norm Activation Type
        """
        super(MixedScaleDenseNet, self).__init__()

        self.n_classes = n_class

        self.conv_in = nn.Sequential(OrderedDict([("conv", nn.Conv2d(in_channels=3, out_channels=in_chns,
                                                                     kernel_size=7, stride=2,
                                                                     padding=3, bias=False)),
                                                  ("norm", norm_act(in_chns)),
                                                  ("pool", nn.MaxPool2d(3, stride=2, padding=1))]))

        self.dense = DenseModule(in_chns, squeeze_ratio, out_chns, num_layers,
                                 dilate_sec=dilate_sec, norm_act=norm_act)

        self.last_channel = self.dense.out_channels  # in_chns + num_layers * out_chns

        # Pooling and predictor
        self.feat_out = norm_act(self.last_channel)
        self.out_se = nn.Sequential(SCSEBlock(channel=self.last_channel, reduction=16))

        if self.n_classes != 0:
            self.aspp = nn.Sequential(ASPPInPlaceABNBlock(self.last_channel, self.last_channel,
                                                          feat_res=(int(in_size[0] / 4), int(in_size[1] / 4)),
                                                          aspp_sec=aspp_sec, norm_act=norm_act))

            self.score_se = nn.Sequential(SCSEBlock(channel=self.last_channel, reduction=16))
            self.score = nn.Sequential(OrderedDict([("norm.1", norm_act(self.last_channel)),
                                                    ("conv.1", nn.Conv2d(self.last_channel, self.last_channel,
                                                                         kernel_size=3, stride=1, padding=2,
                                                                         dilation=2, bias=False)),
                                                    ("norm.2", norm_act(self.last_channel)),
                                                    ("conv.2", nn.Conv2d(self.last_channel, self.n_classes,
                                                                         kernel_size=1, stride=1, padding=0,
                                                                         bias=True)),
                                                    ("up1", nn.Upsample(size=in_size, mode='bilinear'))]))

    def forward(self, x):
        # [N, 3, H, W] -> [N, 32, H/4, W/4] -> [N, 128+32, H/4, W/4]  1/4
        x = self.out_se(self.feat_out(self.dense(self.conv_in(x))))

        if self.n_classes != 0:
            return self.score(self.score_se(self.aspp(x)[1]))
        else:
            return x
