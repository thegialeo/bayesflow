"""Scoring rules for point estimation."""

from .scores import (
    ScoringRule,
    ParametricDistributionRule,
    NormedDifferenceScore,
    MedianScore,
    MeanScore,
    QuantileScore,
    MultivariateNormalScore,
)

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
