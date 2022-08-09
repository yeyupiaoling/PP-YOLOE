import numpy as np

import paddle

__all__ = [
    'normal_',
    'constant_',
    'bias_init_with_prob',
]


def _no_grad_normal_(tensor, mean=0., std=1.):
    with paddle.no_grad():
        tensor.set_value(paddle.normal(mean=mean, std=std, shape=tensor.shape))
    return tensor


def _no_grad_fill_(tensor, value=0.):
    with paddle.no_grad():
        tensor.set_value(paddle.full_like(tensor, value, dtype=tensor.dtype))
    return tensor


def normal_(tensor, mean=0., std=1.):
    """
    Modified tensor inspace using normal_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        mean (float|int): mean value.
        std (float|int): std value.
    Return:
        tensor
    """
    return _no_grad_normal_(tensor, mean, std)


def constant_(tensor, value=0.):
    """
    Modified tensor inspace using constant_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        value (float|int): value to fill tensor.
    Return:
        tensor
    """
    return _no_grad_fill_(tensor, value)


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init
