import numpy as np
from keras import ops

from sklearn.calibration import calibration_curve


def expected_calibration_error(m_true, m_pred, num_bins=10):
    """Estimates the expected calibration error (ECE) of a model comparison network according to [1].

    [1] Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015).
        Obtaining well calibrated probabilities using bayesian binning.
        In Proceedings of the AAAI conference on artificial intelligence (Vol. 29, No. 1).

    Notes
    -----
    Make sure that ``m_true`` are **one-hot encoded** classes!

    Parameters
    ----------
    m_true      : array of shape (num_sim, num_models)
        The one-hot-encoded true model indices.
    m_pred      : array of shape (num_sim, num_models)
        The predicted posterior model probabilities.
    num_bins    : int, optional, default: 10
        The number of bins to use for the calibration curves (and marginal histograms).

    Returns
    -------
    cal_errs    : list of length (num_models)
        The ECEs for each model.
    probs       : list of length (num_models)
        The bin information for constructing the calibration curves.
        Each list contains two arrays of length (num_bins) with the predicted and true probabilities for each bin.
    """

    # Convert tensors to numpy, if passed
    m_true = ops.convert_to_numpy(m_true)
    m_pred = ops.convert_to_numpy(m_pred)

    # Extract number of models and prepare containers
    n_models = m_true.shape[1]
    cal_errs = []
    probs_true = []
    probs_pred = []

    # Loop for each model and compute calibration errs per bin
    for k in range(n_models):
        y_true = (m_true.argmax(axis=1) == k).astype(np.float32)
        y_prob = m_pred[:, k]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=num_bins)

        # Compute ECE by weighting bin errors by bin size
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        binids = np.searchsorted(bins[1:-1], y_prob)
        bin_total = np.bincount(binids, minlength=len(bins))
        nonzero = bin_total != 0
        cal_err = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))

        cal_errs.append(cal_err)
        probs_true.append(prob_true)
        probs_pred.append(prob_pred)
    return cal_errs, probs_true, probs_pred
