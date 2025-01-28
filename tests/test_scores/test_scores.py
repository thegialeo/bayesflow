import keras
import pytest


def test_require_argument_k():
    from bayesflow.scores import NormedDifferenceScore

    with pytest.raises(TypeError) as excinfo:
        NormedDifferenceScore()

    assert "missing 1 required positional argument: 'k'" in str(excinfo)


def test_score_output(basic_scoring_rule, reference):
    target_shape = (reference.shape[0], *basic_scoring_rule.target_shape, reference.shape[-1])
    target = keras.ops.zeros(target_shape)
    score = basic_scoring_rule.score(reference, target)

    assert score.ndim == 0


def test_mean_score_optimality(mean_score, reference):
    suboptimal_target = keras.ops.expand_dims(keras.random.uniform(reference.shape), axis=1)
    optimal_target = keras.ops.expand_dims(reference, axis=1)

    suboptimal_score = mean_score.score(reference, suboptimal_target)
    optimal_score = mean_score.score(reference, optimal_target)

    assert suboptimal_score > optimal_score
    assert keras.ops.isclose(optimal_score, 0)


def test_multivariate_normal_score_output(multivariate_normal_score, reference, mvn_target):
    score = multivariate_normal_score.score(reference, mvn_target)

    assert score.ndim == 0
