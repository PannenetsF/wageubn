from collections import namedtuple
import torch.nn as nn
from .utils import _isinstance

Config = namedtuple('config', [
    'iostrict', 'conv_and_linear_weight', 'conv_and_linear_bias',
    'conv_and_linear_in', 'conv_and_linear_out', 'bn_weight', 'bn_bias',
    'bn_mean', 'bn_var', 'bn_in', 'bn_out', 'acti_in', 'acti'
],
                    defaults=[False, 8, 8, 8, 12, 8, 8, 8, 8, 12, 8, 8, 8])


def config_bn(proc, config, bn_list):
    if _isinstance(proc, bn_list):
        proc.weight_bit_width = config.bn_weight
        proc.bias_bit_width = config.bn_bias
        proc.mean_bit_width = config.bn_mean
        proc.var_bit_width = config.bn_var
        proc.bn_bit_width = config.bn_out
        proc.input_bit_width = config.bn_in


def config_acti(proc, config, acti_list):
    if _isinstance(proc, acti_list):
        proc.acti_bit_width = config.acti
        proc.input_bit_width = config.acti_in


def config_conv_linear(proc, config, conv_linear_list):
    if _isinstance(proc, conv_linear_list):
        proc.weight_bit_width = config.conv_and_linear_weight
        proc.bias_bit_width = config.conv_and_linear_bias
        proc.input_bit_width = config.conv_and_linear_in
        proc.output_bit_width = config.conv_and_linear_out


def config_io(proc, config):
    proc.iostrict = config.iostrict


def config_network(
        net,
        name,
        config,
        bn_list=[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d],
        acti_list=[nn.ReLU, nn.ReLU6],
        conv_linear_list=[nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear],
        show=False):
    proc_list = list(net._modules.keys())
    if proc_list == []:
        if show:
            print(name, 'is configured')
        config_io(net, config)
        config_acti(net, config, acti_list=acti_list)
        config_conv_linear(net, config, conv_linear_list=conv_linear_list)
        config_bn(net, config, bn_list=bn_list)
    else:
        for n in proc_list:
            config_network(net._modules[n],
                           name + '.' + n,
                           config,
                           bn_list=bn_list,
                           acti_list=acti_list,
                           conv_linear_list=conv_linear_list,
                           show=show)
