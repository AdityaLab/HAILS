import numpy as np


# Baseclass for all transforms
class Transform:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reverse(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Normalize(Transform):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def reverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean
