"""
Provide quantilized form of torch.nn.modules.linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number import directquant, alldirectquant


class Linear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_dec_bit_width=8,
                 bias_dec_bit_width=16,
                 input_dec_bit_width=8,
                 output_dec_bit_width=12,
                 weight_all_bit_width=8,
                 bias_all_bit_width=16,
                 input_all_bit_width=8,
                 output_all_bit_width=12,
                 iostrict=False):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_all_bit_width = weight_all_bit_width
        self.bias_all_bit_width = bias_all_bit_width
        self.input_all_bit_width = input_all_bit_width
        self.output_all_bit_width = self.output_all_bit_width
        self.weight_dec_bit_width = weight_dec_bit_width
        self.bias_dec_bit_width = bias_dec_bit_width
        self.input_dec_bit_width = input_dec_bit_width
        self.output_dec_bit_width = self.output_dec_bit_width
        self.iostrict = iostrict

    def linear_forward(self, input):
        if self.iostrict is True:
            input = alldirectquant(input, self.input_dec_bit_width, self.input_all_bit_width)
        if self.bias is None:
            bias = None
        else:
            bias = alldirectquant(self.bias, self.bias_dec_bit_width, self.bias_all_bit_width)
        weight = alldirectquant(self.weight, self.weight_dec_bit_width, self.weight_all_bit_width)
        output = F.linear(input, weight, bias)
        if self.iostrict is True:
            output = alldirectquant(output, self.output_dec_bit_width, self.output_all_bit_width)
        return output

    def forward(self, input):
        return self.linear_forward(input)


if __name__ == '__main__':
    lin = Linear(3, 5)
    x = torch.rand(3, 3)
    print(lin(x).shape)
