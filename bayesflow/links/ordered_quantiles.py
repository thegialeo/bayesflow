import keras

from bayesflow.utils import keras_kwargs

from collections.abc import Sequence

from .ordered import Ordered


class OrderedQuantiles(Ordered):
    def __init__(self, q: Sequence[float] = None, axis: int = None, **kwargs):
        super().__init__(axis, None, **keras_kwargs(kwargs))
        self.q = q

    def build(self, input_shape):
        if self.axis is None and 1 < len(input_shape) <= 3:
            self.axis = -2
        elif self.axis is None:
            raise AssertionError(
                f"Cannot resolve which axis should be ordered automatically from input shape {input_shape}."
            )

        if self.q is None:
            # choose the middle of the specified axis as anchor index
            num_quantile_levels = input_shape[self.axis]
            self.anchor_index = num_quantile_levels // 2
        else:
            # choose quantile level closest to median as anchor index
            self.anchor_index = keras.ops.argmin(keras.ops.abs(keras.ops.convert_to_tensor(self.q) - 0.5))
            assert input_shape[self.axis] == len(self.q)

        super().build(input_shape)
