import torch
import torchvision
from torch import nn, Tensor
import torchvision.transforms as T
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
import os
import sys

configfile = "../../residuals.py"
sys.path.append(os.path.dirname(os.path.expanduser(configfile)))
from residuals import resnet50, resnet34, resnet18, resnet101

class BackBone(nn.Module):
    def __init__(self, images, min_size, max_size, netWork = resnet50(), target: List[Dict[str, Tensor]]= None):
        super().__init__()
        self.netWork = netWork
        self.target = target
        self.min_size = min_size
        self.max_size = max_size

    def forward(self, img, target = None):
        img = [i for i in img]
        if target in not None:
            target_copy: List[Dict[str, Tensor]] = []
            for t in target:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                target_copy.append(data)
            targets = target_copy

        for i in range(len(img)):
            image = img[i]
            target = targets[i] if targets is not None else None

            image, target = self.resize_image_and_mask(image,target)           
            img[i] = image
            if targets is not None and target is not None:
                targets[i] = target



    def resize_image_and_mask(img: Tensor, img_min_size: float, img_max_size: float, target: Optional[Dict[str, Tensor]] = None):

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

