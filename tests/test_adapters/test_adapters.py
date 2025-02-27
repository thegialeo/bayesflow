from keras.saving import (
    deserialize_keras_object as deserialize,
    serialize_keras_object as serialize,
)
import numpy as np


def test_cycle_consistency(adapter, random_data):
    processed = adapter(random_data)
    deprocessed = adapter(processed, inverse=True)

    for key, value in random_data.items():
        if key in ["d1", "d2"]:
            # dropped
            continue
        assert key in deprocessed
        assert np.allclose(value, deprocessed[key])


def test_serialize_deserialize(adapter, custom_objects, random_data):
    processed = adapter(random_data)
    serialized = serialize(adapter)
    deserialized = deserialize(serialized, custom_objects)
    reserialized = serialize(deserialized)

    assert reserialized.keys() == serialized.keys()
    for key in reserialized:
        assert reserialized[key] == serialized[key]

    random_data["foo"] = random_data["x1"]
    deserialized_processed = deserialized(random_data)
    for key, value in processed.items():
        assert np.allclose(value, deserialized_processed[key])


def test_constrain():
    import numpy as np
    import warnings
    from bayesflow.adapters import Adapter

    data = {
        "x1": np.random.exponential(1, size=(32, 1)),
        "x2": -np.random.exponential(1, size=(32, 1)),
        "x3": np.random.beta(0.5, 0.5, size=(32, 1)),
        "x4": np.vstack((np.zeros(shape=(16, 1)), np.ones(shape=(16, 1)))),
        "x5": np.zeros(shape=(32, 1)),
        "x6": np.zeros(shape=(32, 1)),
    }

    adapter = (
        Adapter()
        .constrain("x1", lower=0)
        .constrain("x2", upper=0)
        .constrain("x3", lower=0, upper=1)
        .constrain("x4", lower=0, upper=1, inclusive="both")
        .constrain("x5", lower=0, inclusive="none")
        .constrain("x6", upper=0, inclusive="none")
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = adapter(data)

    # checks if transformations indeed have been applied
    assert result["x1"].min() < 0.0
    assert result["x2"].max() > 0.0
    assert result["x3"].min() < 0.0
    assert result["x3"].max() > 1.0
    assert np.isfinite(result["x4"].min())
    assert np.isfinite(result["x4"].max())
    assert np.isneginf(result["x5"][0])
    assert np.isinf(result["x6"][0])
