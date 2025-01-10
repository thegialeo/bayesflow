import keras

from bayesflow.utils import keras_kwargs

from collections.abc import Sequence


class OrderedQuantiles(keras.Layer):
    def __init__(self, quantile_levels: Sequence[float] = None, axis: int = None, **kwargs):
        super().__init__(**keras_kwargs(kwargs))
        self.quantile_levels = quantile_levels
        self.axis = axis

    def build(self, input_shape):
        super().build(input_shape)
        if 1 < len(input_shape) <= 3:
            self.axis = -2
            if self.quantile_levels is not None:
                num_quantile_levels = len(self.quantile_levels)
                # choose quantile level closest to median as anchor
                self.anchor_quantile_index = keras.ops.argmin(
                    keras.ops.abs(keras.ops.convert_to_tensor(self.quantile_levels) - 0.5)
                )
            else:
                num_quantile_levels = input_shape[self.axis]
                self.anchor_quantile_index = num_quantile_levels // 2

            self.group_indeces = dict(
                below=list(range(0, self.anchor_quantile_index)),
                above=list(range(self.anchor_quantile_index + 1, num_quantile_levels)),
            )
        else:
            raise AssertionError(
                "Cannot resolve which axis should be ordered automatically from input shape " + str(input_shape)
            )

    def call(self, inputs):
        # Divide in anchor, below and above
        below_inputs = keras.ops.take(inputs, self.group_indeces["below"], axis=self.axis)
        anchor_input = keras.ops.take(inputs, self.anchor_quantile_index, axis=self.axis)
        above_inputs = keras.ops.take(inputs, self.group_indeces["above"], axis=self.axis)

        # prepare a reshape target to aid broadcasting correctly
        broadcast_shape = list(below_inputs.shape)  # convert to list to allow item assignment
        broadcast_shape[self.axis] = 1
        broadcast_shape = tuple(broadcast_shape)

        anchor_input = keras.ops.reshape(anchor_input, broadcast_shape)

        # Apply softplus for positivity and cumulate to ensure ordered quantiles
        below = keras.activations.softplus(below_inputs)
        above = keras.activations.softplus(above_inputs)

        below = anchor_input - keras.ops.flip(keras.ops.cumsum(below, axis=self.axis), self.axis)
        above = anchor_input + keras.ops.cumsum(above, axis=self.axis)

        # Concatenate and reshape back
        x = keras.ops.concatenate([below, anchor_input, above], self.axis)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
