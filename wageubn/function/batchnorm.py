"""
Provide quantilized form of torch.nn.modules.batchnorm 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number import directquant


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 weight_bit_width=8,
                 bias_bit_width=8,
                 mean_bit_width=8,
                 var_bit_width=8,
                 bn_bit_width=8,
                 input_bit_width=8,
                 iostrict=False):
        super().__init__(num_features, eps, momentum, affine,
                         track_running_stats)

        self.weight_bit_width = weight_bit_width
        self.bias_bit_width = bias_bit_width
        self.mean_bit_width = mean_bit_width
        self.var_bit_width = var_bit_width
        self.bn_bit_width = bn_bit_width
        self.input_bit_width = input_bit_width
        self.iostrict = iostrict

    def bn_forward(self, input):
        if self.iostrict is True:
            input = directquant(input, self.input_bit_width)
        mean = directquant(self.running_mean, self.mean_bit_width)
        var = directquant(self.running_var, self.var_bit_width)
        if self.affine is True:
            weight = directquant(self.weight, self.weight_bit_width)
            bias = directquant(self.bias, self.bias_bit_width)
            output = directquant(
                F.batch_norm(input,
                             running_mean=mean,
                             running_var=var,
                             weight=weight,
                             bias=bias), self.bn_bit_width)
        else:
            output = directquant(
                F.batch_norm(input, running_mean=mean, running_var=var),
                self.bn_bit_width)
        return output

    def forward(self, input):
        return self.bn_forward(input)
