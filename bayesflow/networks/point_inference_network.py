import keras

from bayesflow.types import Shape, Tensor


class PointInferenceNetwork(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        pass

    def call(
        self,
        xz: Tensor,
        conditions: Tensor = None,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return self._forward(xz, conditions=conditions, training=training, **kwargs)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        raise NotImplementedError

    def compute_metrics(self, x: Tensor, conditions: Tensor = None, stage: str = "training") -> dict[str, Tensor]:
        if not self.built:
            xz_shape = keras.ops.shape(x)
            conditions_shape = None if conditions is None else keras.ops.shape(conditions)
            self.build(xz_shape, conditions_shape=conditions_shape)

        metrics = {}

        if stage != "training" and any(self.metrics):
            # compute sample-based metrics
            # samples = self.sample((keras.ops.shape(x)[0],), conditions=conditions)
            #
            # for metric in self.metrics:
            #     metrics[metric.name] = metric(samples, x)
            pass
            # TODO: instead compute estimate based metrics

        return metrics

    def estimate(self, conditions: Tensor = None) -> Tensor:
        return self._forward(None, conditions)
