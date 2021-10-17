import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from functools import partial
import torch.nn.functional as F
from typing import Callable, Any, Optional, Tuple, List, Dict

class BasicInception(nn.Module):
    def __init__(self,in_channels,num_classes,train=True):
        super().__init__()
        self.stem = stem(in_channels=in_channels)
        self.inception = IncelptionBlock(train=train,num_classes=num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception(x)
        return x

class Conv2dAuto(nn.Conv2d):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2) 

# create instance for Conv2dAuto Class
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

# create convnet with batchnorm
def Conv2d_bn(in_channels, out_channels, conv=conv3x3, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels,eps=0.001))

class stem(nn.Module):
    def __init__(self,in_channels=3,conv_block=Conv2d_bn):
        super().__init__()
        self.Conv2d_1 = conv_block(in_channels, 64, kernel_size= 7,stride=2)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
        self.localrespnorm_1 = nn.LocalResponseNorm(2)
        self.Conv2d_2 = conv_block(64,64, kernel_size=(1,1), stride=1)         
        self.Conv2d_3 = conv_block(64, 192, kernel_size=(3,3), stride=1)
        self.localrespnorm_2 = nn.LocalResponseNorm(2)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)

    def forward(self, x):
        x = self.Conv2d_1(x)
        x = self.maxpool_1(x)
        x = self.localrespnorm_1(x)
        x = self.Conv2d_2(x)
        x = self.Conv2d_3(x)
        x = self.localrespnorm_2(x)
        x = self.maxpool_2(x)
        return x

class AxilaryClassifier(nn.Module):
    def __init__(self,in_features,n_classes,conv_block=Conv2d_bn,*args,**kwargs):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((5,5))
        self.Conv2d_a =  conv_block(in_features,128, kernel_size=(1,1), stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc_1 = nn.Linear(128*5*5,1024)
        self.fc_2 = nn.Linear(1024,n_classes) 
    
    def forward(self, x):
        x = self.avgpool(x)
        x = self.Conv2d_a(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x

class MainClassifier(nn.Module):
    def __init__(self,n_classes, *args, **kwargs):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.dropout = nn.Dropout(p=0.2)
        self.fc_1 = nn.Linear(1024*7*7, 1000)
        self.fc_2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x

class IncelptionBlock(nn.Module):
    def __init__(self, num_classes=1000, train=True, axilary_classifier=AxilaryClassifier, main_classifier=MainClassifier,*args,**kwargs):
        super().__init__()
        self.num_classes=1000
        inception_blocks = [
            InceptionA, InceptionB, InceptionC,InceptionD,
            InceptionE,InceptionF,InceptionG,InceptionH,InceptionI
        ]
        self.train = train
        
        inception_a = inception_blocks[0]
        inception_b = inception_blocks[1]
        inception_c = inception_blocks[2]
        inception_d = inception_blocks[3]
        inception_e = inception_blocks[4]
        inception_f = inception_blocks[5]
        inception_g = inception_blocks[6]
        inception_h = inception_blocks[7]
        inception_i = inception_blocks[8]

        self.inception_a = inception_a(192)
        self.inception_b = inception_b(256)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
        self.inception_c = inception_c(480)
        self.inception_d = inception_d(512)
        self.inception_e = inception_e(512)
        self.inception_f = inception_f(512)
        self.inception_g = inception_g(528)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
        self.inception_h = inception_h(832)
        self.inception_i = inception_i(832)
        self.axilary_classifier_1 = axilary_classifier(in_features=512,n_classes=num_classes)
        self.axilary_classifier_2 = axilary_classifier(in_features=528,n_classes=num_classes)
        self.main_classifier = main_classifier(n_classes=num_classes)


    def forward(self, x):

        x = self.inception_a(x)
        x = self.inception_b(x)
        x = self.maxpool_1(x)
        x = self.inception_c(x)
        if self.train==True:
            ax_out_1 = self.axilary_classifier_1(x) 
        x = self.inception_d(x)
        x = self.inception_e(x)
        x = self.inception_f(x)
        if self.train==True:
            ax_out_2 = self.axilary_classifier_2(x)
        x = self.inception_g(x)
        x = self.maxpool_2(x)
        x = self.inception_h(x)
        x = self.inception_i(x)
        out = self.main_classifier(x)
        if self.train==True:
            return ax_out_1,ax_out_2,out
        else:
            return out


class InceptionA(nn.Module):
    def __init__(self, in_channels=192,conv_block=Conv2d_bn, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.branch1 = conv_block(in_channels, 64, kernel_size=(1,1), stride=1)
        self.branch2 = nn.Sequential(
                    conv_block(in_channels, 96, kernel_size=(1,1), stride=1),
                    conv_block(96, 128, kernel_size=(3,3), stride=1)
        )
        self.branch3 = nn.Sequential(
                    conv_block(in_channels, 16, kernel_size=(1,1), stride=1),
                    conv_block(16, 32, kernel_size=(5,5), stride=1)
        )
        self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1,padding=1,ceil_mode=True),
                    conv_block(in_channels, 32, kernel_size=(1, 1), stride=1)
        )

    def _forward(self, x):

        out_1 = self.branch1(x)
        out_2 = self.branch2(x)
        out_3 = self.branch3(x)
        out_4 = self.branch4(x)

        output = [out_1,out_2,out_3,out_4]
        return output

    def forward(self, x):
        output = self._forward(x)
        return torch.cat(output, 1)

class InceptionB(nn.Module):
    def __init__(self, in_channels=256,conv_block=Conv2d_bn, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.branch1 = conv_block(in_channels, 128, kernel_size=(1, 1), stride=1)
        self.branch2 = nn.Sequential(
                    conv_block(in_channels, 128, kernel_size=(1,1), stride=1),
                    conv_block(128, 192, kernel_size=(3,3), stride=1)
        ) 
        self.branch3 = nn.Sequential(
                    conv_block(in_channels, 32, kernel_size=(1, 1), stride=1),
                    conv_block(32, 96, kernel_size=(5, 5), stride=1)
        )
        self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1,padding=1,ceil_mode=True),
                    conv_block(in_channels, 64, kernel_size=(1, 1), stride=1)
        )

    def _forward(self, x):
        out_1 = self.branch1(x)
        out_2 = self.branch2(x)
        out_3 = self.branch3(x)
        out_4 = self.branch4(x)

        output = [out_1,out_2,out_3,out_4]
        return output

    def forward(self, x):
        output = self._forward(x)
        return torch.cat(output, 1)

class InceptionC(nn.Module):
    def __init__(self,in_channels=480, conv_block=Conv2d_bn, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.branch1 = conv_block(in_channels, 192, kernel_size=(1, 1), stride=1)
        self.branch2 = nn.Sequential(
                    conv_block(in_channels, 96, kernel_size=(1, 1), stride=1),
                    conv_block(96, 208, kernel_size=(3, 3), stride=1)
        )
        self.branch3 = nn.Sequential(
                    conv_block(in_channels, 16, kernel_size=(1, 1), stride=1),
                    conv_block(16, 48, kernel_size=(5, 5), stride=1)
        )
        self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1,padding=1,ceil_mode=True),
                    conv_block(in_channels, 64, kernel_size=(1, 1), stride=1)
        )

    def _forward(self, x):
        out_1 = self.branch1(x)
        out_2 = self.branch2(x)
        out_3 = self.branch3(x)
        out_4 = self.branch4(x)

        output = [out_1,out_2,out_3,out_4]
        return output

    def forward(self, x):
        output = self._forward(x)
        return torch.cat(output, 1)

class InceptionD(nn.Module):
    def __init__(self,in_channels=512, conv_block=Conv2d_bn, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.branch1 = conv_block(in_channels, 160, kernel_size=(1, 1), stride=1)
        self.branch2 = nn.Sequential(
                    conv_block(in_channels, 112, kernel_size=(1, 1), stride=1),
                    conv_block(112, 224, kernel_size=(3, 3), stride=1)
        )
        self.branch3 = nn.Sequential(
                    conv_block(in_channels, 24, kernel_size=(1, 1), stride=1),
                    conv_block(24, 64, kernel_size=(5, 5), stride=1)
        )
        self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1,padding=1,ceil_mode=True),
                    conv_block(in_channels, 64, kernel_size=(1, 1), stride=1)
        )

    def _forward(self, x):
        out_1 = self.branch1(x)
        out_2 = self.branch2(x)
        out_3 = self.branch3(x)
        out_4 = self.branch4(x)

        output = [out_1,out_2,out_3,out_4]
        return output

    def forward(self, x):
        output = self._forward(x)
        return torch.cat(output, 1)

class InceptionE(nn.Module):
    def __init__(self,in_channels=512, conv_block=Conv2d_bn, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.branch1 = conv_block(in_channels, 128, kernel_size=(1, 1), stride=1)
        self.branch2 = nn.Sequential(
                    conv_block(in_channels, 128, kernel_size=(1, 1), stride=1),
                    conv_block(128, 256, kernel_size=(3, 3), stride=1)
        )
        self.branch3 = nn.Sequential(
                    conv_block(in_channels, 24, kernel_size=(1, 1), stride=1),
                    conv_block(24, 64, kernel_size=(5, 5), stride=1)
        )
        self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1,padding=1,ceil_mode=True),
                    conv_block(in_channels, 64, kernel_size=(1, 1), stride=1)
        )

    def _forward(self, x):
        out_1 = self.branch1(x)
        out_2 = self.branch2(x)
        out_3 = self.branch3(x)
        out_4 = self.branch4(x)

        output = [out_1,out_2,out_3,out_4]
        return output

    def forward(self, x):
        output = self._forward(x)
        return torch.cat(output, 1)

class InceptionF(nn.Module):
    def __init__(self,in_channels=512, conv_block=Conv2d_bn, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.branch1 = conv_block(in_channels, 112, kernel_size=(1, 1), stride=1)
        self.branch2 = nn.Sequential(
                    conv_block(in_channels, 144, kernel_size=(1, 1), stride=1),
                    conv_block(144, 288, kernel_size=(3, 3), stride=1)
        )
        self.branch3 = nn.Sequential(
                    conv_block(in_channels, 32, kernel_size=(1, 1), stride=1),
                    conv_block(32, 64, kernel_size=(5, 5), stride=1)
        )
        self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1,padding=1,ceil_mode=True),
                    conv_block(in_channels, 64, kernel_size=(1, 1), stride=1)
        )

    def _forward(self, x):
        out_1 = self.branch1(x)
        out_2 = self.branch2(x)
        out_3 = self.branch3(x)
        out_4 = self.branch4(x)

        output = [out_1,out_2,out_3,out_4]
        return output

    def forward(self, x):
        output = self._forward(x)
        return torch.cat(output, 1)

class InceptionG(nn.Module):
    def __init__(self,in_channels=528, conv_block=Conv2d_bn, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.branch1 = conv_block(in_channels, 256, kernel_size=(1, 1), stride=1)
        self.branch2 = nn.Sequential(
                    conv_block(in_channels, 160, kernel_size=(1, 1), stride=1),
                    conv_block(160, 320, kernel_size=(3, 3), stride=1)
        )
        self.branch3 = nn.Sequential(
                    conv_block(in_channels, 32, kernel_size=(1, 1), stride=1),
                    conv_block(32, 128, kernel_size=(5, 5), stride=1)
        )
        self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1,padding=1,ceil_mode=True),
                    conv_block(in_channels, 128, kernel_size=(1, 1), stride=1)
        )

    def _forward(self, x):
        out_1 = self.branch1(x)
        out_2 = self.branch2(x)
        out_3 = self.branch3(x)
        out_4 = self.branch4(x)

        output = [out_1,out_2,out_3,out_4]
        return output

    def forward(self, x):
        output = self._forward(x)
        return torch.cat(output, 1)

class InceptionH(nn.Module):
    def __init__(self,in_channels=832, conv_block=Conv2d_bn, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.branch1 = conv_block(in_channels, 256, kernel_size=(1, 1), stride=1)
        self.branch2 = nn.Sequential(
                    conv_block(in_channels, 160, kernel_size=(1, 1), stride=1),
                    conv_block(160, 320, kernel_size=(3, 3), stride=1)
        )
        self.branch3 = nn.Sequential(
                    conv_block(in_channels, 32, kernel_size=(1, 1), stride=1),
                    conv_block(32, 128, kernel_size=(5, 5), stride=1)
        )
        self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1,padding=1, ceil_mode=True),
                    conv_block(in_channels, 128, kernel_size=(1, 1), stride=1)
        )

    def _forward(self, x):
        out_1 = self.branch1(x)
        out_2 = self.branch2(x)
        out_3 = self.branch3(x)
        out_4 = self.branch4(x)

        output = [out_1,out_2,out_3,out_4]
        return output

    def forward(self, x):
        output = self._forward(x)
        return torch.cat(output, 1)


class InceptionI(nn.Module):
    def __init__(self,in_channels=832, conv_block=Conv2d_bn, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.branch1 = conv_block(in_channels, 384, kernel_size=(1, 1), stride=1)
        self.branch2 = nn.Sequential(
                    conv_block(in_channels, 192, kernel_size=(1, 1), stride=1),
                    conv_block(192, 384, kernel_size=(3, 3), stride=1)
        )
        self.branch3 = nn.Sequential(
                    conv_block(in_channels, 48, kernel_size=(1, 1), stride=1),
                    conv_block(48, 128, kernel_size=(5, 5), stride=1)
        )
        self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1,padding=1, ceil_mode=True),
                    conv_block(in_channels, 128, kernel_size=(1, 1), stride=1)
        )

    def _forward(self, x):
        out_1 = self.branch1(x)
        out_2 = self.branch2(x)
        out_3 = self.branch3(x)
        out_4 = self.branch4(x)

        output = [out_1,out_2,out_3,out_4]
        return output

    def forward(self, x):
        output = self._forward(x)
        return torch.cat(output, 1)


def GoogleNet(in_channels=3, num_classes=100,train=False):
    return BasicInception(in_channels=in_channels,num_classes=num_classes,train=train)



