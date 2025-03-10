import numpy as np

from collections.abc import Sequence

from .elementwise_transform import ElementwiseTransform


class Sqrt(ElementwiseTransform):
    """Square-root transform a variable.

    Examples
    --------
    >>> adapter = bf.Adapter().sqrt(["x"])
    """

    def __init__(self, keys: Sequence[str]):
        super().__init__()
        self.keys = keys

    def forward(self, data: dict[str, any], **kwargs) -> dict[str, np.ndarray]:
        return {k: (np.sqrt(v) if k in self.keys else v) for k, v in data.items()}

    def inverse(self, data: dict[str, any], **kwargs) -> dict[str, np.ndarray]:
        return {k: (np.square(v) if k in self.keys else v) for k, v in data.items()}

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Sqrt":
        return cls()

    def get_config(self) -> dict:
        return {}
