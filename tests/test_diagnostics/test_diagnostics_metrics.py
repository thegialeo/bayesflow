import numpy as np
import pytest

import bayesflow as bf


def num_variables(x: dict):
    return sum(arr.shape[-1] for arr in x.values())


def test_metric_calibration_error(random_estimates, random_targets, var_names):
    # basic functionality: automatic variable names
    out = bf.diagnostics.metrics.calibration_error(random_estimates, random_targets)
    assert list(out.keys()) == ["values", "metric_name", "variable_names"]
    assert out["values"].shape == (num_variables(random_estimates),)
    assert out["metric_name"] == "Calibration Error"
    assert out["variable_names"] == ["beta_0", "beta_1", "sigma"]

    # user specified variable names
    out = bf.diagnostics.metrics.calibration_error(
        estimates=random_estimates,
        targets=random_targets,
        variable_names=var_names,
    )
    assert out["variable_names"] == var_names

    # user-specifed keys and scalar variable
    out = bf.diagnostics.metrics.calibration_error(
        estimates=random_estimates,
        targets=random_targets,
        variable_keys="sigma",
    )
    assert out["values"].shape == (random_estimates["sigma"].shape[-1],)
    assert out["variable_names"] == ["sigma"]


def test_posterior_contraction(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.metrics.posterior_contraction(random_estimates, random_targets)
    assert list(out.keys()) == ["values", "metric_name", "variable_names"]
    assert out["values"].shape == (num_variables(random_estimates),)
    assert out["metric_name"] == "Posterior Contraction"
    assert out["variable_names"] == ["beta_0", "beta_1", "sigma"]


def test_root_mean_squared_error(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.metrics.root_mean_squared_error(random_estimates, random_targets)
    assert list(out.keys()) == ["values", "metric_name", "variable_names"]
    assert out["values"].shape == (num_variables(random_estimates),)
    assert out["metric_name"] == "NRMSE"
    assert out["variable_names"] == ["beta_0", "beta_1", "sigma"]


def test_expected_calibration_error(pred_models, true_models, model_names):
    out = bf.diagnostics.metrics.expected_calibration_error(pred_models, true_models, model_names=model_names)
    assert list(out.keys()) == ["values", "metric_name", "model_names"]
    assert out["values"].shape == (pred_models.shape[-1],)
    assert out["metric_name"] == "Expected Calibration Error"
    assert out["model_names"] == [r"$\mathcal{M}_0$", r"$\mathcal{M}_1$", r"$\mathcal{M}_2$"]

    # returns probs?
    out = bf.diagnostics.metrics.expected_calibration_error(pred_models, true_models, return_probs=True)
    assert list(out.keys()) == ["values", "metric_name", "model_names", "probs_true", "probs_pred"]
    assert len(out["probs_true"]) == pred_models.shape[-1]
    assert len(out["probs_pred"]) == pred_models.shape[-1]
    # default: auto model names
    assert out["model_names"] == ["M_0", "M_1", "M_2"]

    # handles incorrect input?
    with pytest.raises(Exception):
        out = bf.diagnostics.metrics.expected_calibration_error(pred_models, true_models, model_names=["a"])

    with pytest.raises(Exception):
        out = bf.diagnostics.metrics.expected_calibration_error(pred_models, true_models.transpose)


# -------------------------------------------------------------------------------------------------------------------- #
#                                          Unit tests for MMD Hypothesis Test                                          #
# -------------------------------------------------------------------------------------------------------------------- #


def test_compute_hypothesis_test_from_summaries_shapes():
    """Test the compute_hypothesis_test_from_summaries output shapes."""
    # Mock observed and reference data
    observed_summaries = np.random.rand(10, 5)
    reference_summaries = np.random.rand(100, 5)
    num_null_samples = 50

    mmd_observed, mmd_null = bf.diagnostics.metrics.compute_mmd_hypothesis_test_from_summaries(
        observed_summaries, reference_summaries, num_null_samples=num_null_samples
    )

    # Assertions
    assert isinstance(mmd_observed, float)
    assert isinstance(mmd_null, np.ndarray)
    assert mmd_null.shape == (num_null_samples,)


@pytest.mark.parametrize("summary_network", [lambda data: np.random.rand(data.shape[0], 5), None])
def test_compute_hypothesis_test_shapes(summary_network, monkeypatch):
    """Test the compute_mmd_hypothesis_test output shapes."""
    # Mock observed and reference data
    observed_data = np.random.rand(10, 5)
    reference_data = np.random.rand(100, 5)
    num_null_samples = 50

    # Create a dummy ContinuousApproximator instance
    mock_approximator = bf.approximators.ContinuousApproximator(
        adapter=None,
        inference_network=None,
        summary_network=None,
    )

    # Patch the summary_network attribute of the mock_approximator instance
    monkeypatch.setattr(mock_approximator, "summary_network", summary_network)

    # Call the function under test
    mmd_observed, mmd_null = bf.diagnostics.metrics.compute_mmd_hypothesis_test(
        observed_data, reference_data, mock_approximator, num_null_samples=num_null_samples
    )

    # Assertions
    assert isinstance(mmd_observed, float)
    assert isinstance(mmd_null, np.ndarray)
    assert mmd_null.shape == (num_null_samples,)
