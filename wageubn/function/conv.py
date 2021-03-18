"""
Provide quantilized form of torch.nn.modules.conv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number import directquant, alldirectquant


class Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 weight_dec_bit_width=8,
                 bias_dec_bit_width=16,
                 input_dec_bit_width=8,
                 output_dec_bit_width=12,
                 weight_all_bit_width=8,
                 bias_all_bit_width=16,
                 input_all_bit_width=8,
                 output_all_bit_width=12,
                 iostrict=False):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         padding_mode=padding_mode)
        self.weight_dec_bit_width = weight_dec_bit_width
        self.bias_dec_bit_width = bias_dec_bit_width
        self.input_dec_bit_width = input_dec_bit_width
        self.output_dec_bit_width = output_dec_bit_width
        self.weight_all_bit_width = weight_all_bit_width
        self.bias_all_bit_width = bias_all_bit_width
        self.input_all_bit_width = input_all_bit_width 
        self.output_all_bit_width = output_all_bit_width
        self.iostrict = iostrict
        self.quant = alldirectquant if self.iostrict else directquant

    def conv_forward(self, input):
        if self.iostrict is True:
            input = self.quant(input, self.input_dec_bit_width, self.input_all_bit_width)
        if self.bias is None:
            bias = None
        else:
            bias = self.quant(self.bias, self.bias_dec_bit_width, self.bias_all_bit_width)
        weight = self.quant(self.weight, self.weight_dec_bit_width, self.weight_all_bit_width)
        output = F.conv2d(input, weight, bias, self.stride, self.padding,
                          self.dilation, self.groups)
        if self.iostrict is True:
            output = self.quant(output, self.output_dec_bit_width, self.output_all_bit_width)
        return output

    def forward(self, input):
        return self.conv_forward(input)


if __name__ == '__main__':
    conv = Conv2d(3, 6, 3)
    x = torch.rand(4, 3, 5, 5)
    print(conv(x).shape)
