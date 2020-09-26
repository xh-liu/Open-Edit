import torch
import torch.nn as nn
import functools
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork

class PerturbationNet(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.reg_weight = opt.reg_weight
        self.param1 = nn.Parameter(torch.zeros(1, 1024, 14, 14))
        self.param2 = nn.Parameter(torch.zeros(1, 512, 28, 28))
        self.param3 = nn.Parameter(torch.zeros(1, 256, 56, 56))

    def parameters(self):
        return [self.param1, self.param2, self.param3]

    def forward(self):
        R_reg = torch.pow(self.param1, 2).sum() + torch.pow(self.param2, 2).sum() + torch.pow(self.param3, 2).sum()
        R_reg = R_reg * self.reg_weight
        return [self.param1, self.param2, self.param3], R_reg
