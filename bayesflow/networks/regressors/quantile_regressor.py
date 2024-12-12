from collections.abc import Sequence

import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from bayesflow.utils import find_network, keras_kwargs

from ..point_inference_network import PointInferenceNetwork


@serializable(package="networks.regressors")
class QuantileRegressor(PointInferenceNetwork):
    def __init__(
        self,
        subnet: str | type = "mlp",
        quantile_levels: Sequence[float] = None,
        **kwargs,
    ):
        super().__init__(**keras_kwargs(kwargs))

        if quantile_levels is not None:
            self.quantile_levels = quantile_levels
        else:
            self.quantile_levels = [0.1, 0.9]
        self.quantile_levels = keras.ops.convert_to_tensor(self.quantile_levels)
        self.num_quantiles = len(self.quantile_levels)  # should we have this shorthand?
        # TODO: should we initialize self.num_variables here already? The actual value is assined in build()

        self.body = find_network(subnet, **kwargs.get("subnet_kwargs", {}))
        self.head = keras.layers.Dense(
            units=None, bias_initializer="zeros", kernel_initializer="zeros"
        )  # TODO: why initialize at zero (taken from consistency_model.py)

    # noinspection PyMethodOverriding
    def build(
        self, xz_shape, conditions_shape=None
    ):  # TODO: seems like conditions_shape should definetely be supplied, change to positional argument?
        super().build(xz_shape)

        self.num_variables = xz_shape[-1]
        input_shape = conditions_shape
        self.body.build(input_shape=input_shape)

        input_shape = self.body.compute_output_shape(input_shape)
        self.head.units = self.num_quantiles * self.num_variables
        self.head.build(input_shape=input_shape)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        head_input = self.body(conditions)
        pred_quantiles = self.head(head_input)  # (batch_shape, num_quantiles * num_variables)
        pred_quantiles = keras.ops.reshape(pred_quantiles, (-1, self.num_quantiles, self.num_variables))
        # (batch_shape, num_quantiles, num_variables)

        return pred_quantiles

    def compute_metrics(self, x: Tensor, conditions: Tensor = None, stage: str = "training") -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(x, conditions=conditions, stage=stage)

        true_value = x
        # TODO: keeping like it used to be, but why is do we not set training=(stage=="training") in self.call()
        pred_quantiles = self(x, conditions)
        pointwise_differance = pred_quantiles - true_value[:, None, :]

        loss = pointwise_differance * (
            keras.ops.cast(pointwise_differance > 0, float) - self.quantile_levels[None, :, None]
        )
        loss = keras.ops.mean(loss)

        return base_metrics | {"loss": loss}
