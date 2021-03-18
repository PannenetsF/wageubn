"""
Provide quantilized form of torch.nn.modules.batchnorm 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number import directquant, alldirectquant


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 weight_dec_bit_width=8,
                 bias_dec_bit_width=8,
                 mean_dec_bit_width=8,
                 var_dec_bit_width=8,
                 output_dec_bit_width=8,
                 input_dec_bit_width=8,
                 weight_all_bit_width=8,
                 bias_all_bit_width=8,
                 mean_all_bit_width=8,
                 var_all_bit_width=8,
                 output_all_bit_width=8,
                 input_all_bit_width=8,
                 iostrict=False):
        super().__init__(num_features, eps, momentum, affine,
                         track_running_stats)

        self.weight_dec_bit_width = weight_dec_bit_width
        self.bias_dec_bit_width = bias_dec_bit_width
        self.mean_dec_bit_width = mean_dec_bit_width
        self.var_dec_bit_width = var_dec_bit_width
        self.output_dec_bit_width = output_dec_bit_width
        self.input_dec_bit_width = input_dec_bit_width
        self.weight_all_bit_width = weight_all_bit_width
        self.bias_all_bit_width = bias_all_bit_width
        self.mean_all_bit_width = mean_all_bit_width
        self.var_all_bit_width = var_all_bit_width
        self.output_all_bit_width = output_all_bit_width
        self.input_all_bit_width = input_all_bit_width
        self.iostrict = iostrict
        self.quant = alldirectquant if self.iostrict else directquant

    def bn_forward(self, input):
        if self.iostrict is True:
            input = self.quant(input, self.input_dec_bit_width,
                                   self.input_all_bit_width)
        mean = self.quant(self.running_mean, self.mean_dec_bit_width,
                              self.mean_all_bit_width)
        var = self.quant(self.running_var, self.var_dec_bit_width,
                             self.var_all_bit_width)
        if self.affine is True:
            weight = self.quant(self.weight, self.weight_dec_bit_width,
                                    self.weight_all_bit_width)
            bias = self.quant(self.bias, self.bias_dec_bit_width,
                                  self.bias_all_bit_width)
            output = self.quant(
                F.batch_norm(input,
                             running_mean=mean,
                             running_var=var,
                             weight=weight,
                             bias=bias), self.output_dec_bit_width,
                self.output_all_bit_width)
        else:
            output = self.quant(
                F.batch_norm(input, running_mean=mean, running_var=var),
                self.output_dec_bit_width, self.output_all_bit_width)
        return output

    def forward(self, input):
        return self.bn_forward(input)
