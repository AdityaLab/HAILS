from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        horizon: int,
        lookback: int,
        forecast_start_idx: Optional[int],
        transform=None,
        return_torch: bool = True,
    ):
        self.data = data
        self.horizon = horizon
        self.lookback = lookback
        self.forecast_start_idx = max(lookback, forecast_start_idx or 0)
        self.transform = transform
        self.return_torch = return_torch
        self.X, self.Y = self._create_dataset()

    def _create_dataset(self):
        X, Y = [], []
        for i in range(self.forecast_start_idx, len(self.data) - self.horizon + 1):
            X.append(self.data[i - self.lookback : i])
            Y.append(self.data[i : i + self.horizon])
        return np.array(X), np.array(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.transform:
            x, y = self.transform(x), self.transform(y)
        if self.return_torch:
            x, y = torch.tensor(x), torch.tensor(y)
        return x, y


class HierarchicalTimeSeriesDataset:
    def __init__(
        self,
        data: np.ndarray,
        horizon: int,
        lookback: int,
        forecast_start_idx: Optional[int],
        h_matrix: np.ndarray,
        transform=None,
        return_torch: bool = True,
    ):
        self.time_series_dataset = TimeSeriesDataset(
            data, horizon, lookback, forecast_start_idx, transform, return_torch
        )
        self.h_matrix = h_matrix
