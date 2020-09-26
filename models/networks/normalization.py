import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

## Returns a function that creates a normalization function
## that does not condition on semantic map
def get_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer, opt):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
        else:
            subnorm_type = norm_type

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)        
            
        if subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        elif subnorm_type == 'sync_batch' and opt.mpdist:
            norm_layer = nn.SyncBatchNorm(get_out_channel(layer), affine=True)
        else:
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer
        
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, opt):
        super().__init__()

        if 'instance' in opt.norm_G:
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif 'sync_batch' in opt.norm_G and opt.mpdist:
            self.param_free_norm = nn.SyncBatchNorm(norm_nc, affine=False)
        else:
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)

        nhidden = 128 
        ks = 3
        pw = 1
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, condition):
        normalized = self.param_free_norm(x)

        if x.size()[2] != condition.size()[2]:
            condition = F.interpolate(condition, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(condition)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out
