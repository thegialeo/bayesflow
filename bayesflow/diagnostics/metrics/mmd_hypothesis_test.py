import numpy as np

from bayesflow.approximators import Approximator


def mmd_hypothesis_test_from_summaries(
    observed_summaries: np.ndarray, reference_summaries: np.ndarray
) -> tuple[float, np.ndarray]:
    """Computes the Maximum Mean Discrepancy (MMD) between observed and reference summaries and generates a distribution
    of MMD values under the null hypothesis to assess model fit.

    Parameters
    ----------
    observed_summary : np.ndarray
        Summary statistics of observed data, shape (num_observed, ...).
    reference_summary : np.ndarray
        Summary statistics of reference data, shape (num_reference, ...).

    Returns
    -------
    mmd_observed : float
        The MMD value between observed and reference summaries.
    mmd_null : np.ndarray
        A distribution of MMD values under the null hypothesis.
    """
    pass


def mmd_hypothesis_test(
    observed_data: np.ndarray, reference_data: np.ndarray, approximator: Approximator
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

    Returns
    -------
    mmd_observed : float
        The MMD value between observed and reference data.
    mmd_null : np.ndarray
        A distribution of MMD values under the null hypothesis.
    """
    pass
