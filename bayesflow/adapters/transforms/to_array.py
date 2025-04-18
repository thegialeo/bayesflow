from numbers import Number

import numpy as np
from keras.saving import register_keras_serializable as serializable

from bayesflow.utils.io import deserialize_type, serialize_type
from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.adapters")
class ToArray(ElementwiseTransform):
    """
    Checks provided data for any non-arrays and converts them to numpy arrays.

    This ensures all data is in a format suitable for training.

    Examples
    --------
    >>> ta = bf.adapters.transforms.ToArray()
    >>> a = [1, 2, 3, 4]
    >>> ta.forward(a)
        array([1, 2, 3, 4])
    >>> b = [[1, 2], [3, 4]]
    >>> ta.forward(b)
        array([[1, 2],
            [3, 4]])
    """

    def __init__(self):
        super().__init__()
        self.original_type = None

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ToArray":
        instance = cls()
        instance.original_type = deserialize_type(config["original_type"])
        return instance

    def get_config(self) -> dict:
        return {"original_type": serialize_type(self.original_type)}

    def forward(self, data: any, **kwargs) -> np.ndarray:
        if self.original_type is None:
            self.original_type = type(data)

        return np.asarray(data)

    def inverse(self, data: np.ndarray, **kwargs) -> any:
        if self.original_type is None:
            raise RuntimeError("Cannot call `inverse` before calling `forward` at least once.")

        if issubclass(self.original_type, Number):
            try:
                return self.original_type(data.item())
            except ValueError:
                pass

        # cannot invert
        return data
