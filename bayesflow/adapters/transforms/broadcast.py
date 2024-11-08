from collections.abc import Sequence
import numpy as np

from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from .transform import Transform


@serializable(package="bayesflow.adapters")
class Broadcast(Transform):
    """
    Broadcasts arrays or scalars to the shape of a given other array. Only batch dimensions
    will be considered per default, i.e., all but the last dimension.
    Examples: #TODO
    """

    def __init__(self, keys: Sequence[str], *, to: str, batch_dims_only: bool = True):
        super().__init__()
        self.keys = keys
        self.to = to
        self.batch_dims_only = batch_dims_only

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Broadcast":
        return cls(
            keys=deserialize(config["keys"], custom_objects),
            to=deserialize(config["to"], custom_objects),
            batch_dims_only=deserialize(config["batch_dims_only"], custom_objects),
        )

    def get_config(self) -> dict:
        return {
            "keys": serialize(self.keys),
            "to": serialize(self.to),
            "batch_dims_only": serialize(self.batch_dims_only),
        }

    # noinspection PyMethodOverriding
    def forward(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        target_shape = data[self.to].shape

        if self.batch_dims_only:
            target_shape = target_shape[:-1] + (1,)

        return {k: (np.broadcast_to(v, target_shape) if k in self.keys else v) for k, v in data.items()}

    # noinspection PyMethodOverriding
    def inverse(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        # TODO
        return data
