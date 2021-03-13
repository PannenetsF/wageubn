# wageubn
wageubn's pytorch implementation.

- [wageubn](#wageubn)
  - [Notice](#notice)
  - [wageubn's modules](#wageubns-modules)
    - [wageubn.function](#wageubnfunction)
- [Contributing](#contributing)
- [Acknowledgment](#acknowledgment)

## Notice 

This repo is based on the same framework as [tqt](https://github.com/PannenetsF/TQT) and focuses on inferrence only. Even the quantized error and gradient could be got from `wageubn.function.errorquant` and `wageubn.function.gradquant`, we will not use them. If they are essential for your training, please fork this repo and wrap the `wageubn.function` modules with it.

## wageubn's modules

### wageubn.function 

`function` is a re-impletement of `torch.nn.modules`. Besides all the args used in the original function, a quantized function get 2 kind of optional arguments: `bit_width` and `retrain`. 

`bit_width` has 2 type: weight/bias or activation. 

If the `retrain` is `True`, the Module will be in Retrain Mode, with the `log2_t` trainable. Else, in Static Mode, the `log2_t` are determined by initialization and not trainable.

# Contributing 

It will be great of you to make this project better! There is some ways to contribute!

1. To start with, issues and feature request could let maintainers know what's wrong or anything essential to be added. 
2. If you use the package in you work/repo, just cite the repo and add a dependency note! 
3. You can add some function in `torch.nn` like `HardTanh` and feel free to open a pull request! The code style is simple as [here](style.md).

# Acknowledgment 


The original papar could be find at [Arxiv, Training high-performance and large-scale deep neural networks with full 8-bit integers](https://arxiv.org/abs/1909.02384).