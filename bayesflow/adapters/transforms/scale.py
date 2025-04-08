import numpy as np

from .elementwise_transform import ElementwiseTransform


class Scale(ElementwiseTransform):
    def __init__(self, scale: float | np.ndarray):
        self.scale = scale

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data * self.scale

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data / self.scale
