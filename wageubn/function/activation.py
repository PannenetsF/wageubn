"""
Provide quantilized form of torch.nn.modules.activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number import directquant


class ReLU(nn.ReLU):
    def __init__(self,
                 inplace=False,
                 acti_bit_width=8,
                 retrain=True,
                 quant=False):
        super().__init__(inplace)
        self.acti_bit_width = acti_bit_width

    def relu_forward(self, input):
        return directquant(F.relu(input), self.acti_bit_width)

    def forward(self, input):
        return self.relu_forward(input)


class ReLU6(nn.ReLU6):
    def __init__(self, inplace=False, acti_bit_width=8):
        super().__init__(inplace)
        self.acti_bit_width = acti_bit_width

    def relu6_forward(self, input):
        return directquant(F.relu6(input), self.acti_bit_width)

    def forward(self, input):
        return self.relu6_forward(input)
