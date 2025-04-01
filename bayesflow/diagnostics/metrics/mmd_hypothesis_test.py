"""
This module provides functions for performing hypothesis testing using the Maximum Mean Discrepancy (MMD) metric.

The MMD is a statistical test used to compare two distributions based on their samples. It is commonly used in
machine learning and statistics to assess the similarity between observed data and reference data, or between
summary statistics derived from these datasets.

Functions:
----------
- mmd_hypothesis_test_from_summaries:
    Computes the MMD between observed and reference summaries and generates a null distribution of MMD values
    for hypothesis testing.

- mmd_hypothesis_test:
    Computes the MMD between observed and reference data using an approximator to extract summary statistics,
    and generates a null distribution of MMD values for hypothesis testing.

Dependencies:
-------------
- numpy: For numerical operations.
- bayesflow.approximators: Provides the `Approximator` class for extracting summary statistics.
- bayesflow.metrics: Provides the `maximum_mean_discrepancy` function for computing the MMD.

Usage:
------
These functions can be used to assess the goodness-of-fit of a model by comparing observed data to reference data
or their respective summary statistics.

Example:
--------
# Assuming `observed_data`, `reference_data`, and an `Approximator` instance are available:

This results in namespace collision:
from bayesflow.diagnostics.metrics.mmd_hypothesis_test import mmd_hypothesis_test
from bayesflow.diagnostics.plots.mmd_hypothesis_test import mmd_hypothesis_test

import bayesflow as bf

# Perform the MMD hypothesis test
mmd_observed, mmd_null = bf.diagnostics.metrics.mmd_hypothesis_test(observed_data, reference_data, approximator)

# Plot the null distribution and observed MMD
bf.diagnostics.plots.mmd_hypothesis_test(mmd_null=mmd_null, mmd_observed=mmd_observed)
"""

import numpy as np

from bayesflow.approximators import Approximator
from bayesflow.metrics import maximum_mean_discrepancy


# TODO: maximum_mean_discrepancy expects bayesflow.types.Tensor instead of np.ndarray as input and returns
# bayesflow.types.Tensor instead of float
def mmd_hypothesis_test_from_summaries(
    observed_summaries: np.ndarray,
    reference_summaries: np.ndarray,
    num_null_samples: int = 100,
) -> tuple[float, np.ndarray]:
    """Computes the Maximum Mean Discrepancy (MMD) between observed and reference summaries and generates a distribution
    of MMD values under the null hypothesis to assess model fit.

    Parameters
    ----------
    observed_summary : np.ndarray
        Summary statistics of observed data, shape (num_observed, ...).
    reference_summary : np.ndarray
        Summary statistics of reference data, shape (num_reference, ...).
    num_null_samples : int
        Number of null samples to generate for hypothesis testing. Default is 100.

    Returns
    -------
    mmd_observed : float
        The MMD value between observed and reference summaries.
    mmd_null : np.ndarray
        A distribution of MMD values under the null hypothesis.
    """
    num_observed: int = observed_summaries.shape[0]
    num_reference: int = reference_summaries.shape[0]

    mmd_null_samples: np.ndarray = np.zeros(num_null_samples, dtype=np.float64)

    for i in range(num_null_samples):
        bootstrap_idx: int = np.random.randint(0, num_reference, size=num_observed)
        sampled_summaries: np.ndarray = reference_summaries[bootstrap_idx]
        mmd_null_samples[i] = maximum_mean_discrepancy(
            observed_summaries, sampled_summaries
        )

    mmd_observed: float = maximum_mean_discrepancy(
        observed_summaries, reference_summaries
    )

    return mmd_observed, mmd_null_samples


# TODO: approximator.summary_network takes and returns bayesflow.types.Tensor
def mmd_hypothesis_test(
    observed_data: np.ndarray,
    reference_data: np.ndarray,
    approximator: Approximator,
    num_null_samples: int = 100,
) -> tuple[float, np.ndarray]:
    """Computes the Maximum Mean Discrepancy (MMD) between observed and reference data and generates a distribution of
    MMD values under the null hypothesis to assess model fit.

    Parameters
    ----------
    observed_data : np.ndarray
        Observed data, shape (num_observed, ...).
    reference_data : np.ndarray
        Reference data, shape (num_reference, ...).
    approximator : Approximator
        An instance of the Approximator class used to obtain summary statistics from data.
    num_null_samples : int
        Number of null samples to generate for hypothesis testing. Default is 100.

    Returns
    -------
    mmd_observed : float
        The MMD value between observed and reference data.
    mmd_null : np.ndarray
        A distribution of MMD values under the null hypothesis.
    """
    observed_summaries: np.ndarray = approximator.summary_network(observed_data)
    reference_summaries: np.ndarray = approximator.summary_network(reference_data)

    mmd_observed, mmd_null = mmd_hypothesis_test_from_summaries(
        observed_summaries, reference_summaries, num_null_samples=num_null_samples
    )

    return mmd_observed, mmd_null
