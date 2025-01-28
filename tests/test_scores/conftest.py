import keras
import pytest


@pytest.fixture()
def batch_size():
    return 16


@pytest.fixture()
def num_variables():
    return 4


@pytest.fixture()
def reference(batch_size, num_variables):
    return keras.random.uniform((batch_size, num_variables))


@pytest.fixture()
def median_score():
    from bayesflow.scores import MedianScore

    return MedianScore()


@pytest.fixture()
def mean_score():
    from bayesflow.scores import MeanScore

    return MeanScore()


@pytest.fixture()
def normed_diff_score():
    from bayesflow.scores import NormedDifferenceScore

    return NormedDifferenceScore(k=3)


@pytest.fixture()
def quantile_score():
    from bayesflow.scores import QuantileScore

    return QuantileScore()


@pytest.fixture(params=["median_score", "mean_score", "normed_diff_score", "quantile_score"], scope="function")
def basic_scoring_rule(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def mvn_target(batch_size, num_variables):
    mean_target = keras.ops.zeros((batch_size, 1, num_variables))
    inputs = keras.random.normal((batch_size, num_variables, num_variables))
    print(inputs.shape)
    covariance_target = keras.ops.einsum("...ij,...kj->...ik", inputs, inputs)
    return dict(
        mean=mean_target,
        covariance=covariance_target,
    )


@pytest.fixture()
def multivariate_normal_score(num_variables):
    from bayesflow.scores import MultivariateNormalScore

    return MultivariateNormalScore(D=num_variables)
