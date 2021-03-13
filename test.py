from wageubn.function import BatchNorm2d
import torch

if __name__ == '__main__':
    bn = BatchNorm2d(3)
    x = torch.rand(3, 3, 5, 5)
    print(bn(x).shape)