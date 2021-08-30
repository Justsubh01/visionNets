import torch 
import torch.nn as nn
import torchvision
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VggLayer(nn.Module):
    """
    VggLayer for architecture
    in_linear: Last channel size of feature layer,
    feature: feature layers (Sequencial module)
    num_classes: number of classes for classification task 
    """
    def __init__(self,features,num_classes,*args,**kwargs):
        super(VggLayer, self).__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.decoder = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.decoder(x)

        return x  

def encoder(cfg) -> nn.Sequential:

    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels,cfg[i], kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)]
            in_channels = v
    
    return nn.Sequential(*layers)

cfg = [64, 64, 'M', 128, 128, "M", 256, 256, 256,"M", 512, 512, 512, "M", 512, 512, 512, "M"]

def vggnet(n_classes=1000,features=encoder, cfg=cfg):
    return VggLayer(features(cfg), n_classes)


