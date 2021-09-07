import torch 
import torchvision
import torch.nn as nn
from functools import partial
from torchsummary import summary

# create Conv2d with auto padding
class Conv2dAuto(nn.Conv2d):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2) 

# create instance for Conv2dAuto Class
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

# create dictionaries of activation functions
def activation_func(activation):
    return nn.ModuleDict([
        ["relu", nn.ReLU(inplace=True)],
        ["leaky_relu", nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ["selu", nn.SELU(inplace=True)],
        ["none", nn.Identity()]
    ])[activation]

# create convnet with batchnorm
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))
    
class ResNetResidualBlock(nn.Module):
    """
    Simple Residual network architecture 
    (Main blocks: Contains two convolution layer and a activation function between them,
    Shortcut: When size of previous conv layer output layer and current conv input layer are different
    then we apply shortcut)
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1,conv=conv3x3,activation="relu", *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.downsampling = downsampling
        self.conv = conv

        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )
        self.activate = activation_func(activation)
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)
        ) if self.apply_shortcut else None


    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def apply_shortcut(self):
        return self.in_channels != self.expanded_channels

    def forward(self, x):
        residual = x
        if self.apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels,  block=ResNetResidualBlock, n=1, *args, **kwargs):
        super().__init__()
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class Encoder(nn.Module):
    """
    in_channels: input for 1st layer of encoder, 
    block_size: lenths for layers channels,
    """
    def __init__(self, in_channels=3, blocks_size=[64, 128, 256, 512], deepths = [2,2,2,2],
                activation="relu", block=ResNetResidualBlock,*args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_size

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(self.blocks_sizes, self.blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(self.blocks_sizes[0], self.blocks_sizes[0], n=deepths[0], activation=activation,
                block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                        out_channels, n=n, activation=activation,
                        block=block, *args, **kwargs)
                        for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x
    
class Decoder(nn.Module):

    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)
    
    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):

    def __init__(self, in_channels, n_classes, *args,**kwargs):
        super().__init__()
        self.encoder = Encoder(in_channels, *args, **kwargs)
        self.decoder = Decoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def resnet18(in_channels, n_classes, block=ResNetResidualBlock,*args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[2,2,2,2], *args, **kwargs)

def resnet34(in_channels,n_classes, block=ResNetResidualBlock,*args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3,4,6,3], *args, **kwargs)

def resnet50(in_channels,n_classes, block=ResNetResidualBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3,4,6,3], *args, **kwargs)

def resnet101(in_channels, n_classes, block=ResNetResidualBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3,4,23,3], *args, **kwargs)

def resnet152(in_channels, n_classes, block=ResNetResidualBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 8, 36, 3], *args, **kwargs)




