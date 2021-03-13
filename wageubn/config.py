from collections import namedtuple
import torch.nn as nn
from .utils import _isinstance

Config = namedtuple('config', [
    'conv_and_linear_weight', 'conv_and_linear_bias', 'bn_weight', 'bn_bias',
    'bn_mean', 'bn_var', 'bn_out', 'acti'
],
                    defaults=[8, 8, 8, 8, 8, 8, 8, 8])

bn_list = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
acti_list = [nn.ReLU, nn.ReLU6]
conv_linear_list = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]


def config_bn(proc, config):
    if _isinstance(proc, bn_list):
        proc.weight_bit_width = config.bn_weight
        proc.bias_bit_width = config.bn_bias
        proc.mean_bit_width = config.bn_mean
        proc.var_bit_width = config.bn_var
        proc.bn_bit_width = config.bn_out


def config_acti(proc, config):
    if _isinstance(proc, acti_list):
        proc.acti_bit_width = config.acti


def config_conv_linear(proc, config):
    if _isinstance(proc, conv_linear_list):
        proc.weight_bit_width = config.conv_and_linear_weight
        proc.bias_bit_width = config.conv_and_linear_bias


def config_network(net, name, config, show=False):
    proc_list = list(net._modules.keys())
    if proc_list == []:
        if show:
            print(name, 'is configured')
        config_acti(net, config)
        config_conv_linear(net, config)
        config_bn(net, config)
    else:
        for n in proc_list:
            config_network(net._modules[n], name + '.' + n, config, show=show)
