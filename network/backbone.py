import torch.nn as nn
import torch
from network.resnet import *
from network.head import *
import math




class ByolBackBone(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = resnet50()
        self.head = ByolHead(dim_mlp=2048, dim_out=dim)
    
    def forward(self, x):
        feat = self.net(x)
        embedding = self.head(feat)
        return embedding


