import numpy as np

from bayesflow.approximators import Approximator
from bayesflow.metrics import maximum_mean_discrepancy


def mmd_hypothesis_test_from_summaries(
    observed_summaries: np.ndarray,
    reference_summaries: np.ndarray,
    num_null_samples: int,
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
