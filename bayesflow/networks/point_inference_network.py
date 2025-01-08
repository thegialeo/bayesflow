import keras

from math import prod

from collections.abc import Callable

from bayesflow.utils import keras_kwargs, find_network
from bayesflow.types import Shape, Tensor
from bayesflow.scoring_rules import ScoringRule

# TODO:
# * [ ] weight initialization
# * [ ] serializable ?
# * [ ] testing
# * [ ] docstrings


class PointInferenceNetwork(keras.Layer):
    def __init__(
        self,
        scoring_rules: dict[str, ScoringRule],
        body_subnet: str | type = "mlp",  # naming: shared_subnet / body / subnet ?
        heads_subnet: dict[str, str | keras.Layer] = None,  # TODO: `type` instead of `keras.Layer` ? Too specific ?
        activations: dict[str, keras.layers.Activation | Callable | str] = None,
        **kwargs,
    ):
        super().__init__(
            **keras_kwargs(kwargs)
        )  # TODO: need for bf.utils.keras_kwargs in regular InferenceNetwork class? seems to be a bug

        self.scoring_rules = scoring_rules
        # For now PointInferenceNetwork uses the same scoring rules for all parameters
        # To support using different sets of scoring rules for different parameter (blocks),
        # we can look into renaming this class to sth like `HeadCollection` and
        # handle the split in a higher-level object. (PointApproximator?)

        self.body_subnet = find_network(body_subnet, **kwargs.get("body_subnet_kwargs", {}))

        if heads_subnet:
            self.heads = {
                key: [find_network(value, **kwargs.get("heads_subnet_kwargs", {}).get(key, {}))]
                for key, value in heads_subnet.items()
            }
        else:
            self.heads = {key: [] for key in self.scoring_rules.keys()}

        if activations:
            self.activations = {
                key: (value if isinstance(value, keras.layers.Activation) else keras.layers.Activation(value))
                for key, value in activations.items()
            }  # make sure that each value is an Activation object
        else:
            self.activations = {key: keras.layers.Activation("linear") for key in self.scoring_rules.keys()}
            # TODO: Stefan suggested to call these link functions, decide on this

        for key in self.heads.keys():
            self.heads[key] += [
                keras.layers.Dense(units=None),
                keras.layers.Reshape(target_shape=(None,)),
                self.activations[key],
            ]

        # TODO: allow key-wise overriding of the default, instead of just complete default or totally custom choices

        assert set(self.scoring_rules.keys()) == set(self.heads.keys()) == set(self.activations.keys())

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        # build the shared body network
        input_shape = conditions_shape
        self.body_subnet.build(input_shape)
        body_output_shape = self.body_subnet.compute_output_shape(input_shape)

        for key in self.heads.keys():
            # head_output_shape (excluding batch_size) convention is (*prediction_shape, *parameter_block_shape)
            prediction_shape = self.scoring_rules[key].prediction_shape
            head_output_shape = prediction_shape + xz_shape[1:]

            # set correct head shape
            self.heads[key][-3].units = prod(head_output_shape)
            self.heads[key][-2].target_shape = head_output_shape

            # build head block by block
            input_shape = body_output_shape
            for head_block in self.heads[key]:
                head_block.build(input_shape)
                input_shape = head_block.compute_output_shape(input_shape)

    def call(
        self,
        xz: Tensor,
        conditions: Tensor = None,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # TODO: remove unnecessary simularity with InferenceNetwork
        return self._forward(xz, conditions=conditions, training=training, **kwargs)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        body_output = self.body_subnet(conditions)

        output = dict()
        for key, head in self.heads.items():
            y = body_output
            for head_block in head:
                y = head_block(y)

            output |= {key: y}
        return output

    def compute_metrics(self, x: Tensor, conditions: Tensor = None, stage: str = "training") -> dict[str, Tensor]:
        if not self.built:
            xz_shape = keras.ops.shape(x)
            conditions_shape = None if conditions is None else keras.ops.shape(conditions)
            self.build(xz_shape, conditions_shape=conditions_shape)

        output = self(x, conditions)

        # calculate negative score as mean over all heads
        neg_score = 0
        for key, rule in self.scoring_rules.items():
            neg_score += rule.score(output[key], x)
        neg_score /= len(self.scoring_rules)

        metrics = {}

        if stage != "training" and any(self.metrics):
            # compute sample-based metrics
            # samples = self.sample((keras.ops.shape(x)[0],), conditions=conditions)
            #
            # for metric in self.metrics:
            #     metrics[metric.name] = metric(samples, x)
            pass
            # TODO: instead compute estimate based metrics

        return metrics | {"loss": neg_score}

    def estimate(self, conditions: Tensor = None) -> Tensor:
        return self._forward(None, conditions)
