from collections.abc import Callable, Sequence

from bayesflow.types import Tensor

import keras


class ScoringRule:
    def __init__(
        self,
        name: str = None,
    ):
        self.name = name  # TODO: names for scoring rules may be unnecessary ?

    def score(self, target, reference):
        raise NotImplementedError


class NormedDifferenceLoss(ScoringRule):
    def __init__(
        self,
        k: int = 2,  # results in an estimator for the mean
        name: str = "normed_difference",
    ):
        super().__init__(name)

        self.k = k
        self.target_shape = (1,)

    def score(self, target: Tensor, reference: Tensor) -> Tensor:
        pointwise_differance = target - reference[:, None, :]
        score = keras.ops.absolute(pointwise_differance) ** self.k
        score = keras.ops.mean(score)
        return score


class WeightedNormedDifferenceLoss(ScoringRule):
    def __init__(
        self,
        weighting_function: Callable,
        k: int = 2,
        name: str = "weighted_normed_difference",
    ):
        super().__init__(name)

        if weighting_function:
            self.weighting_function = weighting_function
        else:
            self.weighting_function = lambda input: 1
        self.k = k
        self.target_shape = (1,)

    def score(self, target: Tensor, reference: Tensor) -> Tensor:
        pointwise_differance = target - reference[:, None, :]
        score = self.weighting_function(reference) * keras.ops.absolute(pointwise_differance) ** self.k
        score = keras.ops.mean(score)
        return score


class QuantileLoss(ScoringRule):
    def __init__(
        self,
        quantile_levels: Sequence[float] = [0.1, 0.5, 0.9],
        name: str = "quantile",
    ):
        super().__init__(name)
        self.quantile_levels = keras.ops.convert_to_tensor(quantile_levels)
        self.target_shape = (len(self.quantile_levels),)

    def score(self, target: Tensor, reference: Tensor) -> Tensor:
        pointwise_differance = target - reference[:, None, :]

        score = pointwise_differance * (
            keras.ops.cast(pointwise_differance > 0, float) - self.quantile_levels[None, :, None]
        )
        score = keras.ops.mean(score)
        return score
