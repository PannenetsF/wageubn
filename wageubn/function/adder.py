"""
Provide quantilized form of Adder2d, https://arxiv.org/pdf/1912.13200.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from . import extra as ex
from .number import directquant, alldirectquant


class Adder2d(ex.Adder2d):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=False,
                 weight_dec_bit_width=8,
                 bias_dec_bit_width=16,
                 input_dec_bit_width=8,
                 output_dec_bit_width=12,
                 weight_all_bit_width=8,
                 bias_all_bit_width=16,
                 input_all_bit_width=8,
                 output_all_bit_width=12,
                 iostrict=False):
        super().__init__(input_channel,
                         output_channel,
                         kernel_size,
                         stride=stride,
                         padding=padding,
                         bias=bias)
        self.weight_all_bit_width = weight_all_bit_width
        self.bias_all_bit_width = bias_all_bit_width
        self.input_all_bit_width = input_all_bit_width
        self.output_all_bit_width = output_all_bit_width
        self.weight_dec_bit_width = weight_dec_bit_width
        self.bias_dec_bit_width = bias_dec_bit_width
        self.input_dec_bit_width = input_dec_bit_width
        self.output_dec_bit_width = output_dec_bit_width
        self.iostrict = iostrict
        self.quant = alldirectquant if self.iostrict else directquant

    def adder_forward(self, input):
        if self.iostrict is True:
            input = self.quant(input, self.input_dec_bit_width,
                                   self.input_all_bit_width)
        if self.bias is None:
            bias = None
        else:
            bias = self.quant(self.bias, self.bias_dec_bit_width,
                                  self.bias_all_bit_width)
        weight = self.quant(self.weight, self.weight_dec_bit_width,
                                self.weight_all_bit_width)
        output = ex.adder2d_function(input,
                                     weight,
                                     bias,
                                     stride=self.stride,
                                     padding=self.padding)
        if self.iostrict is True:
            output = self.quant(output, self.output_dec_bit_width,
                                    self.output_all_bit_width)
        return output

    def forward(self, input):
        return self.adder_forward(input)


if __name__ == '__main__':
    add = Adder2d(3, 4, 3, bias=True)
    x = torch.rand(10, 3, 10, 10)
    print(add(x).shape)
