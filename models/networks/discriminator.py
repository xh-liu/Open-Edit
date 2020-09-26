import sys
import torch
import re
import torch.nn as nn
from collections import OrderedDict
import os.path
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_norm_layer
import util.util as util

def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class MultiscaleDiscriminator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        opt.num_D = 2
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = NLayerDiscriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    ## Returns list of lists of discriminator outputs.
    ## The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input, semantics=None):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            if semantics is None:
                out = D(input)
            else:
                out = D(input, semantics)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        opt.n_layers_D = 4
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        nf = opt.ndf
        input_nc = 3
        
        norm_layer = get_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw), opt),
                          nn.LeakyReLU(0.2, False)
            ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        ## We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model'+str(n), nn.Sequential(*sequence[n]))
        
    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
