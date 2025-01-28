from collections.abc import Callable, Sequence

from bayesflow.types import Shape, Tensor

from bayesflow.links import OrderedQuantiles, PositiveSemiDefinite

from bayesflow.utils import logging

import keras

import math

PI = keras.ops.convert_to_tensor(math.pi)


class ScoringRule:
    def get_link(self):
        return keras.layers.Activation("linear")

    def build(self, reference_shape: Shape):
        pass

    def score(self, reference, target):
        raise NotImplementedError


class NormedDifferenceScore(ScoringRule):
    def __init__(
        self,
        k: int,  # results in an estimator for the mean
    ):
        self.k = k
        self.target_shape = (1,)

    def score(self, reference: Tensor, target: Tensor) -> Tensor:
        pointwise_differance = target - reference[:, None, :]
        score = keras.ops.absolute(pointwise_differance) ** self.k
        score = keras.ops.mean(score)
        return score


class MedianScore(NormedDifferenceScore):
    def __init__(self):
        super().__init__(k=1)


class MeanScore(NormedDifferenceScore):
    def __init__(self):
        super().__init__(k=2)


class WeightedNormedDifferenceScore(ScoringRule):
    def __init__(
        self,
        weighting_function: Callable,
        k: int = 2,
    ):
        if weighting_function:
            self.weighting_function = weighting_function
        else:
            self.weighting_function = lambda input: 1
        self.k = k
        self.target_shape = (1,)

    def score(self, reference: Tensor, target: Tensor) -> Tensor:
        pointwise_differance = target - reference[:, None, :]
        score = self.weighting_function(reference) * keras.ops.absolute(pointwise_differance) ** self.k
        score = keras.ops.mean(score)
        return score


class QuantileScore(ScoringRule):
    def __init__(
        self,
        q: Sequence[float] = None,
    ):
        if q is None:
            q = [0.1, 0.5, 0.9]
            logging.info(f"QuantileScore was not provided with argument `q`. Using the default quantile levels: {q}.")

        self.q = keras.ops.convert_to_tensor(q)
        self.target_shape = (len(self.q),)

    def get_link(self):
        if self.q is None:
            raise AssertionError("Needs q to construct link")
        else:
            print(self.q)
            return OrderedQuantiles(self.q)

    def score(self, reference: Tensor, target: Tensor) -> Tensor:
        pointwise_differance = target - reference[:, None, :]

        score = pointwise_differance * (keras.ops.cast(pointwise_differance > 0, float) - self.q[None, :, None])
        score = keras.ops.mean(score)
        return score


class ParametricDistributionRule(ScoringRule):
    """
    TODO
    """

    def __init__(self, target_mappings: dict[str, str] = None):
        self.target_mappings = target_mappings
        self.build(
            (
                None,
                0,
            )
        )  # TODO: make less confusing: this currently needs to be called initially AND another time when the
        # scoring rules attributes are updated instead of just once in the end. This is probably not the Keras way.

    def build(self, reference_shape: Shape):
        if self.target_mappings is None:
            self.target_mappings = {key: key for key in self.compute_target_shape().keys()}

        self.target_shape = self.map_target_keys(self.compute_target_shape(), inverse=True)

    def map_target_keys(self, target_dict: dict[str, str], inverse=False):
        if inverse:
            map = {v: k for k, v in self.target_mappings.items()}
        else:
            map = self.target_mappings
        return {map[key]: value for key, value in target_dict.items()}

    def compute_target_shape(self):
        raise NotImplementedError

    def log_prob(self, x, **kwargs):
        raise NotImplementedError

    def score(self, reference: Tensor, target: dict[str, Tensor]) -> Tensor:
        score = -self.log_prob(x=reference, **self.map_target_keys(target))
        score = keras.ops.mean(score)
        return score * 0.01


class MultivariateNormalScore(ParametricDistributionRule):
    def __init__(self, D: int = None, **kwargs):
        super().__init__(**kwargs)
        self.D = D

    def build(self, reference_shape: Shape):
        if reference_shape is None:
            raise AssertionError("Cannot build before setting D.")
        elif isinstance(reference_shape, tuple) and len(reference_shape) == 2:
            self.D = reference_shape[1]
        else:
            raise AssertionError(f"Cannot build from reference_shape {reference_shape}")
        super().build(reference_shape)

    def compute_target_shape(self) -> dict[str, Shape]:
        return dict(mean=(1,), covariance=(self.D,))

    def get_link(self):
        return self.map_target_keys(
            dict(mean=keras.layers.Activation("linear"), covariance=PositiveSemiDefinite()), inverse=True
        )

    def log_prob(self, x: Tensor, mean: Tensor, covariance: Tensor) -> Tensor:
        diff = x[:, None, :] - mean
        inv_covariance = keras.ops.inv(covariance)
        log_det_covariance = keras.ops.slogdet(covariance)[1]  # Only take the log of the determinant part

        # Compute the quadratic term in the exponential of the multivariate Gaussian
        quadratic_term = keras.ops.einsum("...i,...ij,...j->...", diff, inv_covariance, diff)

        # Compute the log probability density
        log_prob = -0.5 * (self.D * keras.ops.log(2 * PI) + log_det_covariance + quadratic_term)

        return log_prob

    def sample(self, sample_size, mean, covariance):
        batch_size, D = mean.shape
        # Ensure covariance is (batch_size, D, D)
        assert covariance.shape == (batch_size, D, D)

        # Use Cholesky decomposition to generate samples
        chol = keras.ops.cholesky(covariance)
        normal_samples = keras.random.normal((batch_size, D, sample_size))
        samples = mean[:, :, None] + keras.ops.einsum("ijk,ikl->ijl", chol, normal_samples)

        return samples
