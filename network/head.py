import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    def __init__(self, net, dim_in=2048, dim_out=1000, fix_bacobone=True):
        super().__init__()
        self.net = net
        self.fc = nn.Linear(dim_in, dim_out)

        self.fix_bacobone = fix_bacobone

        if fix_bacobone:
            for param in self.net.parameters():
                param.requires_grad = False

        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x, norm=False):
        if self.fix_bacobone:
            with torch.no_grad():
                feat = self.net(x)
                if norm:
                    feat = F.normalize(feat)
        else:
            feat = self.net(x)
            if norm:
                feat = F.normalize(feat)
        return self.fc(feat)




class ByolHead(nn.Module):
    def __init__(self, dim_mlp=2048, dim_out=256, hidden_dim=4096):
        super().__init__()
        self.linear1 = nn.Conv2d(dim_mlp, hidden_dim, 1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(True)
        self.linear2 = nn.Conv2d(hidden_dim, dim_out, 1)
        
    def forward(self, x):
        if x.dim() != 4:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.view(x.size(0), -1)
        return x

