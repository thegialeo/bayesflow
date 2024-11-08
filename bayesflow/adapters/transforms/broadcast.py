from collections.abc import Sequence
import numpy as np

from .transform import Transform


class Broadcast(Transform):
    """
    Broadcasts arrays or scalars to the shape of a given other array. Only batch dimensions
    will be considered per default, i.e., all but the last dimension.
    Examples: #TODO
    """

    def __init__(self, keys: Sequence[str], *, to: str, batch_dims_only: bool = True, scalars_to_arrays: bool = True):
        super().__init__()
        self.keys = keys
        self.to = to
        self.batch_dims_only = batch_dims_only
        self.scalars_to_arrays = scalars_to_arrays

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
