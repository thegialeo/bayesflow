import numpy as np
import pytest


@pytest.fixture()
def random_estimates():
    return {
        "beta": np.random.standard_normal(size=(32, 10, 2)),
        "sigma": np.random.standard_normal(size=(32, 10, 1)),
    }


@pytest.fixture()
def random_targets():
    return {
        "beta": np.random.standard_normal(size=(32, 2)),
        "sigma": np.random.standard_normal(size=(32, 1)),
        "y": np.random.standard_normal(size=(32, 3, 1)),
    }


@pytest.fixture()
def random_priors():
    return {
        "beta": np.random.standard_normal(size=(64, 2)),
        "sigma": np.random.standard_normal(size=(64, 1)),
        "y": np.random.standard_normal(size=(64, 3, 1)),
    }
