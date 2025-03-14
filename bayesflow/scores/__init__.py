"""Scoring rules for point estimation."""

from .normed_difference_score import NormedDifferenceScore
from .mean_score import MeanScore
from .median_score import MedianScore
from .quantile_score import QuantileScore
from .multivariate_normal_score import MultivariateNormalScore

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
