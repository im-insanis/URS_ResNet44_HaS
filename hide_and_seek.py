from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import math
import numpy as np
import torch
from typing import Literal
import torch.nn as nn

class HideAndSeek(nn.Module):
    def __init__(self, apply_prob=0.6, hide_prob=0.5, grid_ratio=0.25, mean=[0.485, 0.456, 0.406], value='Z'):
        super(HideAndSeek, self).__init__()
        self.apply_prob=apply_prob
        self.hide_prob = hide_prob
        self.grid_ratio = grid_ratio
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.value = value
        
    def forward(self, img):
        if not self.training:  # 평가 모드에서는 아무 변화도 주지 않고 입력을 그대로 반환
            return img
       
        if random.random()>self.apply_prob:
            return img
        
        n, c, h, w = img.size()
        # Use Python's round function for float values
        h_grid_step = int(round(h * self.grid_ratio))
        w_grid_step = int(round(w * self.grid_ratio))
        
        for y in range(0, h, h_grid_step):
            for x in range(0, w, w_grid_step):
                y_end = min(h, y + h_grid_step)
                x_end = min(w, x + w_grid_step)
                
                if random.random() < self.hide_prob:
                    continue
                
                if self.value == 'M':
                    img[:, :, y:y_end, x:x_end] = self.mean.to(img.device)
                elif self.value == 'R':
                    img[:, :, y:y_end, x:x_end] = torch.rand_like(img[:, :, y:y_end, x:x_end])
                elif self.value == 'Z':
                    img[:, :, y:y_end, x:x_end] = 0
                        
        return img
    
class HideAndSeek4img(object):
    """
    Summary:
        Hide-and-seek augmentaion
    
    """
    def __init__(self, probability = 0.5,grid_ratio=0.25,patch_probabilty=0.5, mean=[0.4914, 0.4822, 0.4465],value:Literal['M','R','Z'] = "Z"):
        self.probability = probability
        self.grid_ratio = grid_ratio
        self.patch_prob = patch_probabilty
        self.mean = torch.tensor(mean).reshape(-1,1,1)
        self.value = value

    def __call__(self,img:torch.Tensor):
        if random.uniform(0,1)>self.probability:
            return img
        img= img.squeeze()
        c,h,w=torch.tensor(img.shape,dtype=torch.int)
        h_grid_step = torch.round(h*0.25).int()
        w_grid_step = torch.round(w*0.25).int()

        for y in range(0,h,h_grid_step):
            for x in range(0,w,w_grid_step):
                y_end = min(h, y+h_grid_step)  
                x_end = min(w, x+w_grid_step)
                if(random.uniform(0,1) >self.patch_prob):
                    continue
                else:
                    if self.value == 'M':
                        img[:,y:y_end,x:x_end]= self.mean
                    elif self.value == 'R':
                        img[:,y:y_end,x:x_end]=torch.rand_like(img[:,y:y_end,x:x_end])
                    elif self.value =='Z':
                        img[:,y:y_end,x:x_end]=0
                
        return img