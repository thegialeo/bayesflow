from collections.abc import Sequence

from bayesflow.types import Shape, Tensor

from bayesflow.links import OrderedQuantiles, PositiveSemiDefinite

from bayesflow.utils import logging, find_network, serialize_value_or_type, deserialize_value_or_type

import keras

import math


class ScoringRule:
    def __init__(
        self,
        subnets: dict[str, str | type] = dict(),
        subnets_kwargs: dict[str, dict] = dict(),
        links: dict[str, str | type] = dict(),
    ):
        self.subnets = subnets
        self.subnets_kwargs = subnets_kwargs
        self.links = links

        self.config = {
            "subnets_kwargs": subnets_kwargs,
        }

    def get_config(self):
        self.config["subnets"] = {
            key: serialize_value_or_type({}, "subnet", subnet) for key, subnet in self.subnets.items()
        }
        self.config["links"] = {key: serialize_value_or_type({}, "link", link) for key, link in self.links.items()}

        return self.config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["subnets"] = {
            key: deserialize_value_or_type(subnet_dict, "subnet")["subnet"]
            for key, subnet_dict in config["subnets"].items()
        }
        config["links"] = {
            key: deserialize_value_or_type(link_dict, "link")["link"] for key, link_dict in config["links"].items()
        }

        return cls(**config)

    def get_target_shapes(self, reference_shape):
        raise NotImplementedError

    def set_target_shapes(self, reference_shape):
        self.target_shapes = self.get_target_shapes(reference_shape)

    def get_subnet(self, key: str):
        if key not in self.subnets.keys():
            return keras.layers.Identity()
        else:
            return find_network(self.subnets[key], **self.subnets_kwargs.get(key, {}))

    def get_link(self, key: str):
        if key not in self.links.keys():
            return keras.layers.Activation("linear")
        elif isinstance(self.links[key], str):
            return keras.layers.Activation(self.links[key])
        else:
            return self.links[key]

    def get_head(self, key: str):
        subnet = self.get_subnet(key)
        target_shape = self.target_shapes[key]
        dense = keras.layers.Dense(units=math.prod(target_shape))
        reshape = keras.layers.Reshape(target_shape=target_shape)
        link = self.get_link(key)
        return keras.Sequential([subnet, dense, reshape, link])

    def score(self, reference: Tensor, target: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError


class NormedDifferenceScore(ScoringRule):
    def __init__(
        self,
        k: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k

        self.config = {
            "k": k,
        }

    def get_target_shapes(self, reference_shape):
        # keras.saving.load_model sometimes passes reference_shape as a list.
        # This is why I force a conversion to tuple here.
        reference_shape = tuple(reference_shape)
        return dict(value=(1,) + reference_shape[1:])

    def score(self, reference: Tensor, target: dict[str, Tensor]) -> Tensor:
        target = target["value"]
        pointwise_differance = target - reference[:, None, :]
        score = keras.ops.absolute(pointwise_differance) ** self.k
        score = keras.ops.mean(score)
        return score

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config


class MedianScore(NormedDifferenceScore):
    def __init__(self, **kwargs):
        super().__init__(k=1, **kwargs)
        self.config = {}


class MeanScore(NormedDifferenceScore):
    def __init__(self, **kwargs):
        super().__init__(k=2, **kwargs)
        self.config = {}


class QuantileScore(ScoringRule):
    def __init__(self, q: Sequence[float] = None, links=dict(value="ordered_quantiles"), **kwargs):
        super().__init__(links=links, **kwargs)
        if q is None:
            q = [0.1, 0.5, 0.9]
            logging.info(f"QuantileScore was not provided with argument `q`. Using the default quantile levels: {q}.")

        self.q = q
        self._q = keras.ops.convert_to_tensor(q, dtype="float32")
        self.links = {
            key: OrderedQuantiles(q=q) if value == "ordered_quantiles" else value for key, value in links.items()
        }

        self.config = {
            "q": q,
        }

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config

    def get_target_shapes(self, reference_shape):
        # keras.saving.load_model sometimes passes reference_shape as a list.
        # This is why I force a conversion to tuple here.
        reference_shape = tuple(reference_shape)
        return dict(value=(len(self.q),) + reference_shape[1:])

    def score(self, reference: Tensor, target: dict[str, Tensor]) -> Tensor:
        target = target["value"]
        pointwise_differance = target - reference[:, None, :]

        score = pointwise_differance * (keras.ops.cast(pointwise_differance > 0, float) - self._q[None, :, None])
        score = keras.ops.mean(score)
        return score


class ParametricDistributionRule(ScoringRule):
    """
    TODO
    """

    def __init__(self, **kwargs):  # , target_mappings: dict[str, str] = None):
        super().__init__(**kwargs)

    def log_prob(self, x, **kwargs):
        raise NotImplementedError

    def sample(self, batch_shape, **kwargs):
        raise NotImplementedError

    def score(self, reference: Tensor, target: dict[str, Tensor]) -> Tensor:
        score = -self.log_prob(x=reference, **target)
        score = keras.ops.mean(score)
        # multipy to mitigate instability due to relatively high values of parametric score
        return score * 0.01


class MultivariateNormalScore(ParametricDistributionRule):
    def __init__(self, D: int = None, links=dict(covariance=PositiveSemiDefinite()), **kwargs):
        super().__init__(links=links, **kwargs)
        self.D = D

        self.config = {
            "D": D,
        }

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config

    def get_target_shapes(self, reference_shape) -> dict[str, Shape]:
        self.D = reference_shape[-1]
        return dict(
            mean=(self.D,),
            covariance=(self.D, self.D),
        )

    def log_prob(self, x: Tensor, mean: Tensor, covariance: Tensor) -> Tensor:
        diff = x[:, None, :] - mean
        inv_covariance = keras.ops.inv(covariance)
        log_det_covariance = keras.ops.slogdet(covariance)[1]  # Only take the log of the determinant part

        # Compute the quadratic term in the exponential of the multivariate Gaussian
        quadratic_term = keras.ops.einsum("...i,...ij,...j->...", diff, inv_covariance, diff)

        # Compute the log probability density
        log_prob = -0.5 * (self.D * keras.ops.log(2 * math.pi) + log_det_covariance + quadratic_term)

        return log_prob

    # WIP: incorrect draft
    def sample(self, batch_shape, mean, covariance):
        batch_size, D = mean.shape
        # Ensure covariance is (batch_size, D, D)
        assert covariance.shape == (batch_size, D, D)

        # Use Cholesky decomposition to generate samples
        chol = keras.ops.cholesky(covariance)
        normal_samples = keras.random.normal((*batch_shape, D))
        samples = mean[:, :, None] + keras.ops.einsum("...ijk,...ikl->...ijl", chol, normal_samples)

        return samples
