"""
Provide quantilized form of torch.nn.modules.conv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number import directquant


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        weight_bit_width=8,
        bias_bit_width=16,
    ):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         padding_mode=padding_mode)
        self.weight_bit_width = weight_bit_width
        self.bias_bit_width = bias_bit_width

    def conv_forward(self, input):
        if self.bias is None:
            bias = None
        else:
            bias = directquant(self.bias, self.bias_bit_width)
        weight = directquant(self.weight, self.weight_bit_width)
        return F.conv2d(input, weight, bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def forward(self, input):
        return self.conv_forward(input)


if __name__ == '__main__':
    conv = Conv2d(3, 6, 3)
    x = torch.rand(4, 3, 5, 5)
    print(conv(x).shape)