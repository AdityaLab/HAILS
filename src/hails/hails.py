from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal, Poisson

from hails.layers import Corem
from hails.losses import jsd_normal, jsd_poisson
from hails.seq_layers import DLinear, NLinear


class HAILS_Univ(nn.Module):
    def __init__(
        self,
        num_nodes,
        seq_len,
        pred_len,
        pred_model: NLinear | DLinear,
        corem_c: int = 5,
    ) -> None:
        super(HAILS_Univ, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.pred_model = pred_model(
            self.seq_len, self.pred_len, self.num_nodes, dim_out=2, individual=True
        )
        self.corem = Corem(self.num_nodes, corem_c)

    def _forward_base(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get base model predictions

        Args:
            x (torch.Tensor): Input time_series of shape [Batch, Seq length, Num Nodes]

        Returns:
            torch.Tensor: Base model predictions of shape [Batch, Pred length, Num Nodes]
        """
        x = self.pred_model(x)  # [Batch, Pred length, Num Nodes*2]
        mu, logstd = x.split(self.num_nodes, dim=-1)
        return mu, logstd

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            x (torch.Tensor): Input time_series of shape [Batch, Seq length, Num Nodes]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predictions and uncertainty
        """
        mu, logstd = self._forward_base(x)
        _, mu_new, logstd_new, _ = self.corem.predict(mu, logstd)
        return mu_new, logstd_new

    def _get_agg_params(
        self, mu: torch.Tensor, variance: torch.Tensor, hmatrix: torch.Tensor
    ):
        mu_1 = torch.einsum("ji,bti->btj", hmatrix, mu)
        variance_2 = torch.einsum("ji,bti->btj", hmatrix, variance)
        return mu_1, variance_2

    def _dch_normal_loss(
        self, mu: torch.Tensor, logstd: torch.Tensor, hmatrix: torch.Tensor
    ) -> torch.Tensor:
        """Compute Distributional Coherency Loss

        Args:
            mu (torch.Tensor): Mean of the prediction [Batch, Time, Num Nodes]
            logstd (torch.Tensor): Log standard deviation of the prediction [Batch, Time, Num Nodes]
            hmatrix (torch.Tensor): HMatrix for the hierarchy [Num Nodes, Num Nodes]

        Returns:
            torch.Tensor: Loss [Batch, Time, Num Nodes]
        """
        variance = torch.exp(logstd) ** 2
        mu_1, variance_2 = self._get_agg_params(mu, variance, hmatrix)
        norm1 = Normal(mu, torch.exp(logstd))
        norm2 = Normal(mu_1, torch.sqrt(variance_2))
        loss = jsd_normal(norm1, norm2)
        return loss

    def _dch_poisson_loss(
        self, rate: torch.Tensor, hmatrix: torch.Tensor
    ) -> torch.Tensor:
        """Compute Distributional Coherency Loss

        Args:
            rate (torch.Tensor): Rate of the prediction [Batch, Time, Num Nodes]
            hmatrix (torch.Tensor): HMatrix for the hierarchy [Num Nodes, Num Nodes]

        Returns:
            torch.Tensor: Loss [Batch, Time, Num Nodes]
        """
        rate_1, _ = self._get_agg_params(rate, rate, hmatrix)
        poiss1 = Poisson(rate)
        poiss2 = Poisson(rate_1)
        loss = jsd_poisson(poiss1, poiss2)
        return loss
