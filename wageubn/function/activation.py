"""
Provide quantilized form of torch.nn.modules.activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number import directquant, alldirectquant


class ReLU(nn.ReLU):
    def __init__(self,
                 inplace=False,
                 input_dec_bit_width=8,
                 acti_dec_bit_width=8,
                 input_all_bit_width=8,
                 acti_all_bit_width=8,
                 iostrict=False):
        super().__init__(inplace)
        self.acti_dec_bit_width = acti_dec_bit_width
        self.input_dec_bit_width = input_dec_bit_width
        self.acti_all_bit_width = acti_all_bit_width
        self.input_all_bit_width = input_all_bit_width
        self.iostrict = iostrict

    def relu_forward(self, input):
        if self.iostrict is True:
            input = alldirectquant(input, self.input_dec_bit_width,
                                   self.input_all_bit_width)
        return alldirectquant(F.relu(input), self.acti_dec_bit_width,
                              self.acti_all_bit_width)

    def forward(self, input):
        return self.relu_forward(input)


class ReLU6(nn.ReLU):
    def __init__(self,
                 inplace=False,
                 input_dec_bit_width=8,
                 acti_dec_bit_width=8,
                 input_all_bit_width=8,
                 acti_all_bit_width=8,
                 iostrict=False):
        super().__init__(inplace)
        self.acti_dec_bit_width = acti_dec_bit_width
        self.input_dec_bit_width = input_dec_bit_width
        self.acti_all_bit_width = acti_all_bit_width
        self.input_all_bit_width = input_all_bit_width
        self.iostrict = iostrict

    def relu6_forward(self, input):
        if self.iostrict is True:
            input = alldirectquant(input, self.input_dec_bit_width,
                                   self.input_all_bit_width)
        return alldirectquant(F.relu6(input), self.acti_dec_bit_width,
                              self.acti_all_bit_width)

    def forward(self, input):
        return self.relu6_forward(input)
