"""
Basic quantization function types are defined for forward propagation and back propagation.
Shown in the paper's Sec. III.C.
"""

import torch
from torch.autograd import Function
import math
from ._utils import number_to_tensor


class StochasticRound(Function):
    @staticmethod
    def forward(ctx, x):
        x_shape = x.shape
        eps = 1e-8
        x = x.reshape(-1, 1)
        choice = torch.cat((x.floor(), x.ceil()), dim=1)
        weight = torch.cat((x.ceil() - x + eps, x + eps - x.floor()), dim=1)
        sample = torch.multinomial(weight, 1)
        idx = torch.cat((sample == 0, sample == 1), dim=1)
        x = choice[idx].reshape(x_shape)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


sround = StochasticRound.apply


class DirectQuant(Function):
    @staticmethod
    def forward(ctx, x, k):
        return torch.clamp(
            torch.round(x * 2**(k - 1)) / 2**(k - 1), -2**(k - 1) + 1,
            2**(k - 1) - 1)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


directquant = DirectQuant.apply


class ConstQuant(Function):
    @staticmethod
    def forward(ctx, x, k_dr, k):
        r = 2**torch.round(x.abs().max().log2())
        norm = x / r
        dr = 2**(k_dr - 1)
        sd = torch.clamp(sround(dr * norm), -dr + 1, dr - 1)
        return sd / 2**(k - 1)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None, None


constquant = ConstQuant.apply


class ShiftQuant(Function):
    @staticmethod
    def forward(ctx, x, k):
        r = 2**torch.round(x.abs().max().log2())
        norm = x / r
        dk = 1 / 2**(k - 1)
        sq = r * torch.clamp(directquant(norm, k), -1 + dk, 1 - dk)
        return sq

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


shiftquant = ShiftQuant.apply


class GradOfWeightQuant(Function):
    @staticmethod
    def forward(ctx, x, kgw, kdr):
        ctx.save_for_backward(torch.tensor(kgw), torch.tensor(kdr))
        return x

    @staticmethod
    def backward(ctx, grad_out):
        kgw, kdr = ctx.saved_tensors
        return constquant(grad_out, kdr, kgw), None, None


gwquant = GradOfWeightQuant.apply


class GradOfBNQuant(Function):
    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(torch.tensor(k))
        return x

    @staticmethod
    def backward(ctx, grad_out):
        k = ctx.saved_tensors[0]
        return directquant(grad_out, k), None


gbnquant = GradOfBNQuant.apply


class ErrorQuant(Function):
    @staticmethod
    def forward(ctx, x, k):
        r = 2**torch.round(x.abs().max().log2())
        sc = r / 2**(k - 1)
        cmp0 = ((x / sc).abs >= 1) * sc * torch.clamp(
            (x / sc).round(), 1 - 2**(k - 1), 2**(k - 1) - 1)
        cmp1 = ((x / sc).abs < 1) * directquant(x / sc, k)
        return cmp1 + cmp0

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


errquant = ErrorQuant.apply