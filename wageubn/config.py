from collections import namedtuple
import torch.nn as nn
from . import function as wnn
from .utils import _isinstance

Config = namedtuple('config', [
    'iostrict', 'conv_and_linear_weight', 'conv_and_linear_bias',
    'conv_and_linear_in', 'conv_and_linear_out', 'bn_weight', 'bn_bias',
    'bn_mean', 'bn_var', 'bn_in', 'bn_out', 'acti_in', 'acti',
    'conv_and_linear_weight_all', 'conv_and_linear_bias_all',
    'conv_and_linear_in_all', 'conv_and_linear_out_all', 'bn_weight_all',
    'bn_bias_all', 'bn_mean_all', 'bn_var_all', 'bn_in_all', 'bn_out_all',
    'acti_in_all', 'acti_all'
],
                    defaults=[
                        False,
                        8,
                        8,
                        8,
                        12,
                        8,
                        8,
                        8,
                        8,
                        12,
                        8,
                        8,
                        8,
                    ])


def config_bn(proc, config, bn_list):
    if _isinstance(proc, bn_list):
        proc.weight_dec_bit_width = config.bn_weight
        proc.bias_dec_bit_width = config.bn_bias
        proc.mean_dec_bit_width = config.bn_mean
        proc.var_dec_bit_width = config.bn_var
        proc.output_dec_bit_width = config.bn_out
        proc.input_dec_bit_width = config.bn_in

        proc.weight_all_bit_width = config.bn_weight_all
        proc.bias_all_bit_width = config.bn_bias_all
        proc.mean_all_bit_width = config.bn_mean_all
        proc.var_all_bit_width = config.bn_var_all
        proc.output_all_bit_width = config.bn_out_all
        proc.input_all_bit_width = config.bn_in_all


def config_acti(proc, config, acti_list):
    if _isinstance(proc, acti_list):
        proc.acti_dec_bit_width = config.acti
        proc.input_dec_bit_width = config.acti_in

        proc.acti_all_bit_width = config.acti_all
        proc.input_all_bit_width = config.acti_in_all


def config_conv_linear(proc, config, conv_linear_list):
    if _isinstance(proc, conv_linear_list):
        proc.weight_dec_bit_width = config.conv_and_linear_weight
        proc.bias_dec_bit_width = config.conv_and_linear_bias
        proc.input_dec_bit_width = config.conv_and_linear_in
        proc.output_dec_bit_width = config.conv_and_linear_out

        proc.weight_all_bit_width = config.conv_and_linear_weight_all
        proc.bias_all_bit_width = config.conv_and_linear_bias_all
        proc.input_all_bit_width = config.conv_and_linear_in_all
        proc.output_all_bit_width = config.conv_and_linear_out_all


def config_io(proc, config):
    proc.iostrict = config.iostrict


def config_network(net,
                   name,
                   config,
                   bn_list=[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d],
                   acti_list=[nn.ReLU, nn.ReLU6],
                   conv_linear_list=[
                       nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, wnn.Adder2d
                   ],
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
