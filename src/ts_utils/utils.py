import random

import numpy as np
import torch as th


def set_seed(seed: int = 42):
    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    return None


def prob_poisson(samples: th.Tensor, sparsity_threshold: float = 0.3):
    """Text if the samples are Poisson distributed

    Args:
        samples (th.Tensor): Samples from the distribution [Samples, Num Nodes]
        sparsity_threshold (float, optional): Threshold for sparsity. Defaults to 0.3.

    Returns:
        th.Tensor: Probability of the samples being Poisson distributed [Num Nodes] (0 if normal, 1 if Poisson)
    """
    # If the samples are sparse, then it is likely to be Poisson
    return ((samples == 0).sum(dim=0) / samples.size(0) < sparsity_threshold).float()


def prob_poisson_dispersion(samples: th.Tensor, disp_threshold: float = 0.2):
    """Text if the samples are Poisson distributed

    Args:
        samples (th.Tensor): Samples from the distribution [Samples, Num Nodes]
        sparsity_threshold (float, optional): Threshold for sparsity. Defaults to 0.3.

    Returns:
        th.Tensor: Probability of the samples being Poisson distributed [Num Nodes]
    """
    # n = samples.size(0)
    sample_mean = samples.mean(dim=0)
    sample_variance = samples.var(dim=0)
    D = sample_variance / sample_mean
    # chi_square_stat = (n - 1) * D / sample_mean
    # res = 1 - 2 * abs((1 - chi2.cdf(sample_variance.numpy(), n - 1)) - 0.5)
    return (((1 - disp_threshold) < D) & (D < (1 + disp_threshold))).float()
