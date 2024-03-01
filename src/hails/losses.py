import torch
from torch.distributions import Normal, Poisson


def jsd_normal(dist1: Normal, dist2: Normal):
    r"""
    Jensen-Shannon Divergence between two normal distributions

    Args:
    - dist1: Normal distribution 1 of shape [batch, latent_dim]
    - dist2: Normal distribution 2 of shape [batch, latent_dim]

    Returns:
    - loss: Jensen-Shannon Divergence of shape [batch, latent_dim]
    """
    mu1, std1 = dist1.loc, dist1.scale  # [batch, latent_dim]
    mu2, std2 = dist2.loc, dist2.scale  # [batch, latent_dim]
    loss = 0.5 * (
        (std1**2 + (mu1 - mu2) ** 2) / (2 * std2**2)
        + (std2**2 + (mu2 - mu1) ** 2) / (2 * std1**2)
        - 1
    )
    return loss  # [batch, latent_dim]


def jsd_poisson(dist1: Poisson, dist2: Poisson):
    r"""
    Jensen-Shannon Divergence between two poisson distributions

    Args:
    - dist1: Poisson distribution 1 of shape [batch, latent_dim]
    - dist2: Poisson distribution 2 of shape [batch, latent_dim]

    Returns:
    - loss: Jensen-Shannon Divergence of shape [batch, latent_dim]
    """
    rate1 = dist1.rate  # [batch, latent_dim]
    rate2 = dist2.rate  # [batch, latent_dim]
    loss = rate1 * (torch.log(rate1) - torch.log(rate2)) + rate2 * (
        torch.log(rate2) - torch.log(rate1)
    )
    return loss  # [batch, latent_dim]
