import math
from typing import List

import torch
import torch.nn as nn
from torch.distributions import Normal


class FFN(nn.Module):
    """
    Generic Feed Forward Network class
    """

    def __init__(
        self,
        in_dim: int,
        hidden_layers: List[int],
        out_dim: int,
        activation=nn.ReLU,
        dropout: float = 0.0,
    ):
        r"""
        ## Inputs
        :param in_dim: Input dimensions
        :param hidden_layers: List of hidden layer sizes
        :param out_dim: Output dimensions
        :param activation: nn Module for activation
        :param Dropout: rate of dropout
        ```python
        print("hello")
        ```
        """
        super(FFN, self).__init__()
        layers = [nn.Linear(in_dim, hidden_layers[0]), activation()]
        for i in range(1, len(hidden_layers)):
            layers.extend(
                [
                    nn.Linear(hidden_layers[i - 1], hidden_layers[i]),
                    activation(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_layers[-1], out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, inp):
        r"""
        ## Inputs

        :param inp: Input vectors shape: [batch, inp_dim]
        ----
        ## Outputs
        out: [batch, out_dim]
        """
        return self.layers(inp)


class LatentEncoder(nn.Module):
    """
    Generic Stochastic Encoder using FFN
    """

    def __init__(
        self,
        in_dim: int,
        hidden_layers: List[int],
        out_dim: int,
        activation=nn.ReLU,
        dropout: float = 0.0,
    ):
        r"""
        ## Inputs
        :param in_dim: Input dimensions
        :param hidden_layers: List of hidden layer sizes
        :param out_dim: Output dimensions
        :param activation: nn Module for activation
        :param Dropout: rate of dropout
        """
        super(LatentEncoder, self).__init__()
        self.out_dim = out_dim
        self.net = FFN(in_dim, hidden_layers, out_dim * 2, activation, dropout)

    def forward(self, inp):
        r"""
        ## Inputs
        :param inp: Input vectors shape: [batch, inp_dim]
        ----
        ## Outputs
        mean: [batch, out_dim]
        logscale: [batch, out_dim],
        dist: Normal Distribution with mean and logscale
        """
        out = self.net(inp)
        mean, logscale = torch.split(out, self.out_dim, dim=-1)
        dist = Normal(mean, logscale)
        out = dist.rsample()
        return out, mean, logscale, dist


class GRUEncoder(nn.Module):
    """
    Encodes Sequences using GRU
    """

    def __init__(self, in_size: int, out_dim: int, bidirectional: bool = False):
        super(GRUEncoder, self).__init__()
        self.out_dim = out_dim // 2 if bidirectional else out_dim
        self.gru = nn.GRU(
            in_size, self.out_dim, batch_first=True, bidirectional=bidirectional
        )

    def forward(self, batch):
        r"""
        ## Inputs
        :param batch: Input vectors shape: [batch, seq_len, in_size]
        ----
        ## Outputs
        out: [batch, seq_len, out_dim]
        """
        out_seq, _ = self.gru(batch)
        return out_seq[:, -1, :]


class Corem(nn.Module):
    def __init__(self, nodes: int, c: float = 5.0) -> None:
        super(Corem, self).__init__()
        self.nodes = nodes
        self.c = c
        self.w_hat = nn.Parameter(torch.randn(self.nodes) / math.sqrt(self.nodes) + c)
        self.w = nn.Linear(self.nodes, self.nodes)
        self.b = nn.Parameter(torch.randn(self.nodes) / math.sqrt(self.nodes) + c)
        self.v1 = nn.Linear(self.nodes, self.nodes)
        self.v2 = nn.Linear(self.nodes, self.nodes)

    def forward(self, mu: torch.Tensor, logstd: torch.Tensor, y: torch.Tensor):
        """
        Args:
        mu: torch.Tensor of shape [batch, nodes]
        logstd: torch.Tensor of shape [batch, nodes]
        y: torch.Tensor of shape [batch, nodes]

        Returns:
        mu_final: torch.Tensor of shape [batch, nodes]
        logstd_final: torch.Tensor of shape [batch, nodes]
        log_pyM: torch.Tensor of shape
        """
        gamma = torch.sigmoid(self.w_hat)  # [nodes]
        mu_final: torch.Tensor = gamma * mu + (1 - gamma) * self.w(mu)  # [batch, nodes]
        logstd_final: torch.Tensor = torch.sigmoid(self.b) * logstd + (
            1.0 - torch.sigmoid(self.b)
        ) * (self.v1(mu) + self.v2(logstd))  # [batch, nodes]
        py = Normal(mu_final, logstd_final)
        log_pyM = torch.sum(py.log_prob(y))  # [batch]
        return mu_final, logstd_final, log_pyM, py

    def predict(self, mu: torch.Tensor, logstd: torch.Tensor, sample=True):
        gamma = torch.sigmoid(self.w_hat)
        mu_final: torch.Tensor = gamma * mu + (1 - gamma) * self.w(mu)
        logstd_final: torch.Tensor = torch.sigmoid(self.b) * logstd + (
            1.0 - torch.sigmoid(self.b)
        ) * (self.v1(mu) + self.v2(logstd))
        py = Normal(mu_final, logstd_final.exp())
        if sample:
            y_new_i = py.sample()
        else:
            y_new_i = mu_final
        return y_new_i, mu_final, logstd_final, py
