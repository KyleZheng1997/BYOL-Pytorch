import torch
import torch.nn as nn
from network.backbone import *
from math import pi, cos


class Byol(nn.Module):
    def __init__(self, dim=256, base_momentum=0.996):
        super().__init__()
        self.encoder_q = ByolBackBone()
        self.encoder_t = ByolBackBone()
        
        self.predictor = ByolHead(dim_mlp=dim, dim_out=dim)

        self.base_momentum = base_momentum

        for param_o, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.requires_grad = False
            param_t.data.copy_(param_o.data)

    @torch.no_grad()
    def _momentum_update_target_encoder(self, cur_iter, max_iter):
        momentum = 1 - (1 - self.base_momentum) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2
        for param_o, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.data = param_t.data * momentum + param_o.data * (1. - momentum)

    def forward(self, x, y, current_iter, max_iter):
        inputs = torch.cat([x, y])
        q = self.predictor(self.encoder_q(inputs))

        with torch.no_grad():
            self._momentum_update_target_encoder(current_iter, max_iter)
            reverse_inputs = torch.cat([y, x])
            t = self.encoder_t(reverse_inputs)
        return F.normalize(q), F.normalize(t)


