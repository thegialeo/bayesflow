import bayesflow as bf


def test_metric_calibration_error(random_estimates, random_targets):
    # basic functionality: automatic variable names
    ce = bf.diagnostics.metrics.calibration_error(random_estimates, random_targets)
    assert list(ce.keys()) == ["values", "metric_name", "variable_names"]
    assert ce["values"].shape == (3,)
    assert ce["metric_name"] == "Calibration Error"
    assert ce["variable_names"] == ["beta_1", "beta_2", "sigma"]

    # user specified variable names
    var_names = [r"$\beta_0$", r"$\beta_1$", r"$\sigma$"]
    ce = bf.diagnostics.metrics.calibration_error(
        estimates=random_estimates, targets=random_targets, variable_names=var_names
    )
    assert ce["variable_names"] == var_names

    # user-specifed keys and scalar variable
    ce = bf.diagnostics.metrics.calibration_error(
        estimates=random_estimates,
        targets=random_targets,
        variable_keys="sigma",
    )
    assert ce["values"].shape == (1,)
    assert ce["variable_names"] == ["sigma"]


def test_posterior_contraction(random_estimates, random_targets):
    # basic functionality: automatic variable names
    ce = bf.diagnostics.metrics.posterior_contraction(random_estimates, random_targets)
    assert list(ce.keys()) == ["values", "metric_name", "variable_names"]
    assert ce["values"].shape == (3,)
    assert ce["metric_name"] == "Posterior Contraction"
    assert ce["variable_names"] == ["beta_1", "beta_2", "sigma"]


def test_root_mean_squared_error(random_estimates, random_targets):
    # basic functionality: automatic variable names
    ce = bf.diagnostics.metrics.root_mean_squared_error(random_estimates, random_targets)
    assert list(ce.keys()) == ["values", "metric_name", "variable_names"]
    assert ce["values"].shape == (3,)
    assert ce["metric_name"] == "NRMSE"
    assert ce["variable_names"] == ["beta_1", "beta_2", "sigma"]
