import importlib
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.loss import *
from models.networks.discriminator import *
from models.networks.generator import *
from models.networks.encoder import *
import util.util as util


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network
    
    return network

def create_network(cls, opt, init=True, cuda=True):
    net = cls(opt)
    net.print_network()
    assert(torch.cuda.is_available())
    if cuda:
        if opt.mpdist:
            net.cuda(opt.gpu)
        else:
            net.cuda()
    if init:
        net.init_weights(opt.init_type, opt.init_variance)
    return net

def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, 'generator')
    return create_network(netG_cls, opt)


def define_D(opt):
    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    return create_network(netD_cls, opt)


def define_E(opt):
    netE_cls = find_network_using_name(opt.netE, 'encoder')
    netE = create_network(netE_cls, opt, init=False, cuda=False)
    state_dict = torch.load(opt.vse_enc_path, map_location='cpu')
    netE.load_state_dict(state_dict['model'][0])
    if opt.mpdist:
        netE.cuda(opt.gpu)
    else:
        netE.cuda()
    netE.eval()
    return netE
