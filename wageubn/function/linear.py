"""
Provide quantilized form of torch.nn.modules.linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number import directquant


class Linear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_bit_width=8,
                 bias_bit_width=16):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_bit_width = weight_bit_width
        self.bias_bit_width = bias_bit_width

    def linear_forward(self, input):
        if self.bias is None:
            bias = None
        else:
            bias = directquant(self.bias, self.bias_bit_width)
        weight = directquant(self.weight, self.weight_bit_width)
        return F.linear(input, weight, bias)

    def forward(self, input):
        return self.linear_forward(input)


if __name__ == '__main__':
    lin = Linear(3, 5)
    x = torch.rand(3, 3)
    print(lin(x).shape)
