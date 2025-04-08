import numpy as np

from .elementwise_transform import ElementwiseTransform

class Shift(ElementwiseTransform):
    def __init__(self, shift: float | np.ndarray):
        self.shift = shift

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data + self.shift

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data - self.shift



