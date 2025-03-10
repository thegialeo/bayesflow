import numpy as np

from collections.abc import Sequence

from .elementwise_transform import ElementwiseTransform


class Log(ElementwiseTransform):
    """Log transforms a variable.

    Parameters
    ----------
    p1 : boolean
        Add 1 to the input before taking the logarithm?

    Examples
    --------
    >>> adapter = bf.Adapter().log(["x"])
    """

    def __init__(self, keys: Sequence[str], *, p1: bool = False):
        super().__init__()
        self.keys = keys
        self.p1 = p1

    def forward(self, data: dict[str, any], **kwargs) -> dict[str, np.ndarray]:
        if self.p1:
            return {k: (np.log1p(v) if k in self.keys else v) for k, v in data.items()}
        else:
            return {k: (np.log(v) if k in self.keys else v) for k, v in data.items()}

    def inverse(self, data: dict[str, any], **kwargs) -> dict[str, np.ndarray]:
        if self.p1:
            return {k: (np.expm1(v) if k in self.keys else v) for k, v in data.items()}
        else:
            return {k: (np.exp(v) if k in self.keys else v) for k, v in data.items()}

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Log":
        return cls()

    def get_config(self) -> dict:
        return {}
