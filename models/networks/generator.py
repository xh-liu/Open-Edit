import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.architecture import SPADEResnetBlock
import random
import numpy as np


class OpenEditGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw = 7
        self.sh = 7

        self.head_0 = SPADEResnetBlock(16*nf, 16*nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16*nf, 16*nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16*nf, 16*nf, opt)

        self.up_0 = SPADEResnetBlock(16*nf, 8*nf, opt)
        self.up_1 = SPADEResnetBlock(8*nf, 4*nf, opt)
        self.up_2 = SPADEResnetBlock(4*nf, 2*nf, opt)
        self.up_3 = SPADEResnetBlock(2*nf, 1*nf, opt)

        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x, edge, perturbation=False, p=None):

        # input: 1024 x 7 x 7
        x = self.head_0(x, edge) # 1024 x 7 x 7

        x = self.up(x) # 1024 x 14 x 14
        x = self.G_middle_0(x, edge) # 1024 x 14 x 14
        if perturbation:
            x = x + p[0]

        x = self.G_middle_1(x, edge) # 1024 x 14 x 14

        x = self.up(x) # 1024 x 28 x 28
        x = self.up_0(x, edge) # 512 x 28 x 28
        if perturbation:
            x = x + p[1]
        x = self.up(x) # 512 x 56 x 56
        x = self.up_1(x, edge)
        if perturbation:
            x = x + p[2]
        x = self.up(x) # 256 x 112 x 112
        x = self.up_2(x, edge)
        x = self.up(x) # 128 x 224 x 224
        x = self.up_3(x, edge)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x

