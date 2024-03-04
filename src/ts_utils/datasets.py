from typing import Optional

import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        horizon: int,
        lookback: int,
        forecast_start_idx: Optional[int],
        scaled: bool = False,
        transform=None,
    ):
        self.data = data
        self.horizon = horizon
        self.lookback = lookback
        self.forecast_start_idx = max(lookback, forecast_start_idx or 0)
        self.transform = transform
        self.X, self.Y = self._create_dataset()
        self.scaled = scaled

    def _create_dataset(self):
        self.means = self.data.mean(dim=0)
        self.stds = self.data.std(dim=0)
        self.maxs = self.data.max(dim=0).values
        X, Y = [], []
        for i in range(self.forecast_start_idx, len(self.data) - self.horizon + 1):
            X.append(self.data[i - self.lookback : i])
            Y.append(self.data[i : i + self.horizon])
        return torch.stack(X), torch.stack(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.transform:
            x, y = self.transform(x), self.transform(y)
        if self.scaled:
            # x = (x - self.means) / self.stds
            # y = (y - self.means) / self.stds
            x = x / self.maxs
            y = y / self.maxs
        return x, y


class HierarchicalTimeSeriesDataset:
    def __init__(
        self,
        data: torch.Tensor,
        horizon: int,
        lookback: int,
        forecast_start_idx: Optional[int],
        h_matrix: torch.Tensor,
        scaled: bool = True,
        transform=None,
    ):
        self.time_series_dataset = TimeSeriesDataset(
            data, horizon, lookback, forecast_start_idx, scaled, transform
        )
        self.h_matrix = h_matrix
