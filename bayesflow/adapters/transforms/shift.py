from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
import numpy as np

from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.adapters")
class Shift(ElementwiseTransform):
    def __init__(self, shift: np.typing.ArrayLike):
        self.shift = np.array(shift)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ElementwiseTransform":
        return cls(shift=deserialize(config["shift"]))

    def get_config(self) -> dict:
        return {"shift": serialize(self.shift)}

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data + self.shift

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data - self.shift
