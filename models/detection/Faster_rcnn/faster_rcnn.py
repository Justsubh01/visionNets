import torch
import torchvision
from torch import nn, Tensor
import torchvision.transforms as T
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
import os
import sys

configfile = "../../"
sys.path.append(os.path.dirname(os.path.expanduser(configfile)))
from vgg16 import vggnet

class BackBone(nn.Module):
    def __init__(self,min_size, max_size, netWork = vggnet(n_classes=1000)):
        super().__init__()
        self.netWork = netWork
        self.min_size = min_size
        self.max_size = max_size

    def forward(self, imgs, target = None):
        imgs = [i for i in imgs]
        if target is not None:
            target_copy: List[Dict[str, Tensor]] = []
            for t in target:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                target_copy.append(data)
            target = target_copy
        
        for i in range(len(imgs)):
            image = imgs[i]
            target = target[i] if target is not None else None
            image, target = self.resize_image_and_mask(image, self.min_size, self.max_size)           
            imgs[i] = image
            if target is not None:
                target[i] = target

        images = torch.stack(imgs, dim=0) 
        req_features = []
        model = self.netWork
        fc = list(model.features)
        k = images
        for i in fc: 
            k = i(k)
            if k.shape[-1] <= self.min_size // 16:
                break
            req_features.append(i)
            out_channels = k.size()[1]

        feature_extractor = nn.Sequential(*req_features)
        image = feature_extractor(images)
        
        return image, target

    def resize_image_and_mask(self,img: Tensor, img_min_size: float, img_max_size: float, target: Optional[Dict[str, Tensor]] = None):

        img_current_shape = torch.tensor(img.shape[-2:])

        min_size = torch.min(img_current_shape).to(dtype=torch.float32)
        max_size = torch.max(img_current_shape).to(dtype=torch.float32)

        scale = torch.min(img_min_size / min_size, img_max_size / max_size)

        scale_factor = scale.item()

        image = F.interpolate(img[None], scale_factor=scale_factor, mode = "bilinear", align_corners=False)[0]

        if target is None:
            return image, target 

        if "masks" in target:
            mask = target["masks"]
            mask = F.interpolate(mask[:, None].float(), scale_factor=scale_factor)[:, 0].byte()

            target["masks"] = mask
        
        if "boxes" in target:
            (h, w) = img.shape[-2:]
            (nw_h, nw_w) = image.shape[-2:]
            bbox = target["boxes"]

            ratio = [
                torch.tensor(nw_s, dtype=torch.float32, device=bbox.device) /
                torch.tensor(o_s, dtype=torch.float32, device=bbox.device)
                for nw_s, o_s in zip((nw_h, nw_w), (h,w))
            ]

            ratio_height, ratio_width = ratio
            xmin, ymin, xmax, ymax = bbox.unbind(1)

            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height

            boxes = torch.stack((xmin, ymin, xmax, ymax), dim=1) 

            target["boxes"] = boxes

        return image,target

# class RPNetwork(nn.Module):


# images = torch.rand(4,3,254, 254)

# model = BackBone(min_size=600, max_size=1000)
# model(images)