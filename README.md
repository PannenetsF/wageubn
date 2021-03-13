# wageubn
wageubn's pytorch implementation.

- [wageubn](#wageubn)
  - [Notice](#notice)
  - [wageubn's modules](#wageubns-modules)
    - [wageubn.function](#wageubnfunction)
    - [wageubn.threshold](#wageubnthreshold)
  - [Build a network with wageubn](#build-a-network-with-wageubn)
  - [Rebuild a network with wageubn](#rebuild-a-network-with-wageubn)
  - [Initialize a network's threshold](#initialize-a-networks-threshold)
  - [Train Something with Pre-Trained Model](#train-something-with-pre-trained-model)
  - [Turn a network to quantized or not](#turn-a-network-to-quantized-or-not)
  - [Exclude some types of module](#exclude-some-types-of-module)
  - [Do analyse over the activations and weights](#do-analyse-over-the-activations-and-weights)
- [Contributing](#contributing)
- [Acknowledgment](#acknowledgment)

## Notice 

This repo is based on the same framework as [tqt](https://github.com/PannenetsF/TQT)

## wageubn's modules

### wageubn.function 

`function` is a re-impletement of `torch.nn.modules`. Besides all the args used in the original function, a quantized function get 2 kind of optional arguments: `bit_width` and `retrain`. 

`bit_width` has 2 type: weight/bias or activation. 

If the `retrain` is `True`, the Module will be in Retrain Mode, with the `log2_t` trainable. Else, in Static Mode, the `log2_t` are determined by initialization and not trainable.

### wageubn.threshold

Provide 3 ways to initialize the threshold: `init_max`, `init_kl_j`, `init_3sd`. 

To initialize the weight and threshold correctly, please follow the method to build a network with wageubn.

## Build a network with wageubn

To get output of each wageubn module, the network should be flat, that is, no `nn.Sequential`, no nested `nn.ModuleList`. 

You'd better use `nn.ModuleList` and append every operation after it. If there're some operations that are `nn.ModuleList` of some operation, you can use `.extend` to keep the network flat. 

## Rebuild a network with wageubn 

Much often we need to re-train a network, and we can do a quick job with `lambda`. As you can see in the file `lenet.py`, with the change of the wrapper, a net could be simply converted into a quantized one. 

## Initialize a network's threshold 

Just 3 steps! 

1. Add hook for output storage.
2. Adjust the threshold via `wageubn.threshold` 
3. Remove hook.

## Train Something with Pre-Trained Model

Supposed that you have a pretrained model, and it's hard to change all keys in its state dictionary. More often, it may contain lots of `nn.Module` but not specially `nn.ModuleList`. A dirty but useful way is simply change the `import torch.nn as nn` to `import wageubn.function as nn`. You can get a quant-style network with all previous keys unchanged! 

All you need to do is add a list `self.proc` to the network module.

Through `wageubn.threshold.add_hook_general`, we can add hook for any network if you add a list containing all operations used in forward.

Let's get some example: 

```py
# noquant.py
import torch.nn as nn 

class myNet(nn.Module):
    def __init__(self, args):
        self.op1 = ... 
        self.op2 = ...
        if args:
            self.op_args = ...
        ...
    def forward(self, x):
        ...
```

and

```py
# quant.py
import wageubn.function as nn 

class myNet(nn.Module):
    def __init__(self):
        self.op1 = ... 
        self.op2 = ...
        if args:
            self.op_args = ...
            self.proc.append('op_args')
        ...
    def forward(self, x):
        ...
```

We can load and retrain by:

```py
# main.py 
import wageubn
from unquant import myNet as oNet
from quant import myNet as qNet

handler = wageubn.threshold.hook_handler

train(oNet) ... 
wageubn.threshold.add_hook(oNet, 'oNet', handler)
qNet.load_state_dict(oNet.state_dict(), strict=False)
for (netproc, qnetproc) in zip(funct_list, qfunct_list):
    wageubn.threshold.init.init_network(netproc, qnetproc, show=True)
retrain(qNet)
```

## Turn a network to quantized or not

With a network built by [method metioned](#train-something-with-pre-trained-model), we may need use a quant/or-not version. So we implement `wageubn.utils.make_net_quant_or_not` to change its mode easily.

## Exclude some types of module

Normally we wil disable the quantization of batchnorm modules, you can simply exclude the bn in `wageubn.utils.make_net_quant_or_not` like:

```py
wageubn.utils.make_net_quant_or_not(net,
                                'net',
                                quant=True,
                                exclude=[torch.nn.BatchNorm2d],
                                show=True)
```

## Do analyse over the activations and weights

Always, we need to do analysis over activations and weights to choose a proper way to quantize the network. We implement some function do these. It's recommend do this with tensorboard.

`wageubn.threshold.get_hook` will get all hook output got from the forward with their module name as a tuple. 

```py
net = QNet()
wageubn.utils.make_net_quant_or_not(net, quant=True)
wageubn.threshold.add_hook(net, 'net', wageubn.threshold.hook_handler)
net.cuda()
for i, (images, labels) in enumerate(data_test_loader):
    net(images.cuda())
    break
out = get_hook(net, 'net', show=True)
for i in out:
    print(i[0], i[1].shape)
writer.add_histogram(i[0], i[1].cpu().data.flatten().detach().numpy())
```

Similarly, the weights could be get from `net.named_parameters()`.


# Contributing 

It will be great of you to make this project better! There is some ways to contribute!

1. To start with, issues and feature request could let maintainers know what's wrong or anything essential to be added. 
2. If you use the package in you work/repo, just cite the repo and add a dependency note! 
3. You can add some function in `torch.nn` like `HardTanh` and feel free to open a pull request! The code style is simple as [here](style.md).

# Acknowledgment 


The original papar could be find at [Arxiv, Training high-performance and large-scale deep neural networks with full 8-bit integers](https://arxiv.org/abs/1909.02384).