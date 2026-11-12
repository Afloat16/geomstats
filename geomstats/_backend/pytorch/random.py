"""Torch based random backend."""

import torch as _torch
from torch import rand, randint
from torch.distributions.multivariate_normal import (
    MultivariateNormal as _MultivariateNormal,
)

from ._dtype import _allow_complex_dtype


def choice(x, a, p=None, replace=True):
    """Generate a random sample from a tensor along axis 0."""
    if not _torch.is_tensor(x):
        return x

    if not replace and not 0 <= a <= len(x):
        raise ValueError("Sample size must be between zero and the population size.")

    if p is None:
        idx = (
            _torch.randint(len(x), (a,), device=x.device)
            if replace
            else _torch.randperm(len(x), device=x.device)[:a]
        )
    else:
        idx = _torch.multinomial(
            _torch.as_tensor(p, device=x.device),
            num_samples=a,
            replacement=replace,
        )

    return x[idx]


def seed(*args, **kwargs):
    return _torch.manual_seed(*args, **kwargs)


@_allow_complex_dtype
def normal(loc=0.0, scale=1.0, size=(1,)):
    if not hasattr(size, "__iter__"):
        size = (size,)
    return _torch.normal(mean=loc, std=scale, size=size)


def uniform(low=0.0, high=1.0, size=(1,), dtype=None):
    if not hasattr(size, "__iter__"):
        size = (size,)
    if low >= high:
        raise ValueError("Upper bound must be higher than lower bound")
    return (high - low) * rand(*size, dtype=dtype) + low


@_allow_complex_dtype
def multivariate_normal(mean, cov, size=(1,)):
    if not hasattr(size, "__iter__"):
        size = (size,)
    return _MultivariateNormal(mean, cov).sample(size)
