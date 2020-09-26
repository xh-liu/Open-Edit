import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from models.networks.base_network import BaseNetwork
from models.networks.bdcn import BDCN_multi
import pdb

def l2norm(x, norm_dim=1):
    norm = torch.pow(x, 2).sum(dim=norm_dim, keepdim=True).sqrt()
    x = torch.div(x, norm)
    return x

class ResNetbdcnEncoder(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        img_dim = 2048
        embed_size = 1024
        self.cnn = self.get_cnn('resnet152', False)
        self.cnn = nn.Sequential(self.cnn.conv1, self.cnn.bn1,
                                 self.cnn.relu, self.cnn.maxpool,
                                 self.cnn.layer1, self.cnn.layer2,
                                 self.cnn.layer3, self.cnn.layer4)
        self.conv1x1 = nn.Conv2d(img_dim, embed_size, 1)
        # bdcn
        self.edgenet = BDCN_multi(level=opt.edge_level)
        edge_params = torch.load(opt.edge_model_path, map_location='cpu')
        self.edgenet.load_state_dict(edge_params)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.tanh = nn.Tanh()
        self.edge_tanh = opt.edge_tanh

    def forward(self, x0, skip=False, norm=True, pool=False):
        xe = (x0 + 1) / 2.0 * 255
        xe = xe[:,[2,1,0],:,:]
        xe[:,0,:,:] = xe[:,0,:,:] - 123.0
        xe[:,1,:,:] = xe[:,1,:,:] - 117.0
        xe[:,2,:,:] = xe[:,2,:,:] - 104.0
        edge = self.edgenet(xe)
        x = self.cnn(x0)
        x = self.conv1x1(x)
        if pool:
            x = self.avg_pool(x)
        if norm:
            x = l2norm(x)
        if self.edge_tanh:
            return x, self.tanh(edge)
        else:
            return x, edge

    def get_cnn(self, arch, pretrained):
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()
        return model

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
