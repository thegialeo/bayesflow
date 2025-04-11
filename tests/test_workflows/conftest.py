import pytest


@pytest.fixture()
def inference_network():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(depth=2)


@pytest.fixture()
def summary_network():
    from bayesflow.networks import TimeSeriesTransformer

    return TimeSeriesTransformer(embed_dims=(8, 8), mlp_widths=(32, 32), mlp_depths=(1, 1))
