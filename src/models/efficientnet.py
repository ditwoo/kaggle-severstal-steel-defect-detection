import torch
import torch.nn as nn
import torch.nn.functional as F


IS_PYTORCH_PAD = True
IS_GATHER_EXCITE = False


def drop_connect(x, probability, training):
    if not training: return x

    batch_size = len(x)
    keep_probability = 1 - probability
    noise = keep_probability
    noise += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    mask = torch.floor(noise)
    x = x / keep_probability * mask

    return x


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):  #MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)


class Identity(nn.Module):
    def forward(self, x):
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(len(x),-1)


class Conv2dBn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, zero_pad=[0,0,0,0], group=1):
        super(Conv2dBn, self).__init__()
        if IS_PYTORCH_PAD: zero_pad = [kernel_size//2]*4
        self.pad  = nn.ZeroPad2d(zero_pad)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=0, stride=stride, groups=group, bias=False)
        self.bn   = nn.BatchNorm2d(out_channel, eps=1e-03, momentum=0.01)
        #print(zero_pad)

    def forward(self, x):
        x = self.pad (x)
        x = self.conv(x)
        x = self.bn  (x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_channel, reduction_channel, excite_size):
        super(SqueezeExcite, self).__init__()
        self.excite_size=excite_size

        self.squeeze = nn.Conv2d(in_channel, reduction_channel, kernel_size=1, padding=0)
        self.excite  = nn.Conv2d(reduction_channel, in_channel, kernel_size=1, padding=0)
        self.act = Swish()


    def forward(self, x):
        if IS_GATHER_EXCITE:
            s = F.avg_pool2d(x, kernel_size=self.excite_size)
        else:
            s = F.adaptive_avg_pool2d(x,1)

        s = self.act(self.squeeze(s))
        s = torch.sigmoid(self.excite(s))

        if IS_GATHER_EXCITE:
            s = F.interpolate(s, size=(x.shape[2],x.shape[3]), mode='nearest')

        x = s*x
        return x


class EfficientBlock(nn.Module):
    def __init__(self, in_channel, channel, out_channel, kernel_size, stride, zero_pad, excite_size, drop_connect_rate):
        super().__init__()
        self.is_shortcut = stride == 1 and in_channel == out_channel
        self.drop_connect_rate = drop_connect_rate

        if in_channel == channel:
            self.bottleneck = nn.Sequential(
                Conv2dBn(   channel, channel, kernel_size=kernel_size, stride=stride, zero_pad=zero_pad, group=channel),
                Swish(),
                SqueezeExcite(channel, in_channel//4, excite_size) if excite_size>0
                else Identity(),
                Conv2dBn(channel, out_channel, kernel_size=1, stride=1),
            )
        else:
            self.bottleneck = nn.Sequential(
                Conv2dBn(in_channel, channel, kernel_size=1, stride=1),
                Swish(),
                Conv2dBn(   channel, channel, kernel_size=kernel_size, stride=stride, zero_pad=zero_pad, group=channel),
                Swish(),
                SqueezeExcite(channel, in_channel//4, excite_size) if excite_size>0
                else Identity(),
                Conv2dBn(channel, out_channel, kernel_size=1, stride=1)
            )

    def forward(self, x):
        b = self.bottleneck(x)
        if self.is_shortcut:
            if self.training: b = drop_connect(b, self.drop_connect_rate, True)
            x = b + x
        else:
            x = b
        return x


class ConvGnUp2d(nn.Module):
    def __init__(self, in_channel, out_channel, num_group=32, kernel_size=3, padding=1, stride=1):
        super(ConvGnUp2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.gn   = nn.GroupNorm(num_group,out_channel)

    def forward(self,x):
        x = self.conv(x)
        x = self.gn(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x


def upsize_add(x, lateral):
    return F.interpolate(x, size=lateral.shape[2:], mode='nearest') + lateral


class EfficientUnet(nn.Module):
    def __init__(self, num_classes: int = 4, drop_connection_rate: float = 0.2):
        super(EfficientUnet, self).__init__()
        # e = EfficientNet.from_pretrained('efficientnet-b5')
        d = drop_connection_rate
        self.stem   = nn.Sequential(
            Conv2dBn(3,48, kernel_size=3,stride=2,zero_pad=[0,1,0,1]),
            Swish()
        )
        self.block1 = nn.Sequential(
            EfficientBlock( 48,  48,  24, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=128, drop_connect_rate=d*1/7),
            * [EfficientBlock( 24,  24,  24, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=128, drop_connect_rate=d*1/7) for i in range(1,3)],
        )
        self.block2 = nn.Sequential(
            EfficientBlock( 24, 144,  40, kernel_size=3, stride=2, zero_pad=[0,1,0,1], excite_size= 64, drop_connect_rate=d*2/7),
            * [EfficientBlock( 40, 240,  40, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size= 64, drop_connect_rate=d*2/7) for i in range(1,5)],
        )
        self.block3 = nn.Sequential(
            EfficientBlock( 40, 240,  64, kernel_size=5, stride=2, zero_pad=[1,2,1,2], excite_size= 32, drop_connect_rate=d*3/7),
            * [EfficientBlock( 64, 384,  64, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 32, drop_connect_rate=d*3/7) for i in range(1,5)],
        )
        self.block4 = nn.Sequential(
            EfficientBlock( 64, 384, 128, kernel_size=3, stride=2, zero_pad=[0,1,0,1], excite_size= 16, drop_connect_rate=d*4/7),
            * [EfficientBlock(128, 768, 128, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size= 16, drop_connect_rate=d*4/7) for i in range(1,7)],
        )
        self.block5 = nn.Sequential(
            EfficientBlock(128, 768, 176, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 16, drop_connect_rate=d*5/7),
            * [EfficientBlock(176,1056, 176, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 16, drop_connect_rate=d*5/7) for i in range(1,7)],
        )
        self.block6 = nn.Sequential(
            EfficientBlock(176,1056, 304, kernel_size=5, stride=2, zero_pad=[1,2,1,2], excite_size=  8, drop_connect_rate=d*6/7),
            * [EfficientBlock(304,1824, 304, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size=  8, drop_connect_rate=d*6/7) for i in range(1,9)],
        )
        self.block7 = nn.Sequential(
            EfficientBlock(304,1824, 512, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=  8, drop_connect_rate=d*7/7),
            * [EfficientBlock(512,3072, 512, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=  8, drop_connect_rate=d*7/7) for i in range(1,3)],
        )

        self.last = nn.Sequential(
            Conv2dBn(512, 2048,kernel_size=1,stride=1),
            Swish()
        )

        self.lateral0 = nn.Conv2d(2048, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral1 = nn.Conv2d( 176, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral2 = nn.Conv2d(  64, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral3 = nn.Conv2d(  40, 64,  kernel_size=1, padding=0, stride=1)

        self.top1 = nn.Sequential(
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
        )
        self.top2 = nn.Sequential(
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
        )
        self.top3 = nn.Sequential(
            ConvGnUp2d( 64, 64),
        )
        self.top4 = nn.Sequential(
            nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.logit_mask = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        batch_size,C,H,W = x.shape

        x = self.stem(x)            #; print('stem  ',x.shape)
        x = self.block1(x)    ;x0=x #; print('block1',x.shape)
        x = self.block2(x)    ;x1=x #; print('block2',x.shape)
        x = self.block3(x)    ;x2=x #; print('block3',x.shape)
        x = self.block4(x)          #; print('block4',x.shape)
        x = self.block5(x)    ;x3=x #; print('block5',x.shape)
        x = self.block6(x)          #; print('block6',x.shape)
        x = self.block7(x)          #; print('block7',x.shape)
        x = self.last(x)      ;x4=x #; print('last  ',x.shape)

        # segment
        t0 = self.lateral0(x4)
        t1 = upsize_add(t0, self.lateral1(x3)) #16x16
        t2 = upsize_add(t1, self.lateral2(x2)) #32x32
        t3 = upsize_add(t2, self.lateral3(x1)) #64x64

        t1 = self.top1(t1) #128x128
        t2 = self.top2(t2) #128x128
        t3 = self.top3(t3) #128x128

        t = torch.cat([t1,t2,t3],1)
        t = self.top4(t)
        logit_mask = self.logit_mask(t)
        logit_mask = F.interpolate(logit_mask, scale_factor=2.0, mode='bilinear', align_corners=False)

        return logit_mask

if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 1600)
    m = EfficientUnet(4)
    out = m(x).shape
    print(out)
