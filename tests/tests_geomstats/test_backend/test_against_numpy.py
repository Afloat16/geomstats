import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.online_kmeans import OnlineKMeans
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.backend import AgainstNumpyTestCase

from .data.against_numpy import AgainstNumpyTestData


class TestAgainstNumpy(AgainstNumpyTestCase, metaclass=DataBasedParametrizer):
    testing_data = AgainstNumpyTestData()


@pytest.mark.parametrize(
    "first_dtype, second_dtype",
    [
        (gs.float32, gs.float64),
        (gs.float64, gs.float32),
        (gs.int32, gs.float64),
        (gs.float64, gs.int64),
        (gs.complex64, gs.complex128),
        (gs.complex64, gs.float64),
        (gs.float64, gs.complex64),
        (gs.int32, gs.int64),
    ],
)
def test_close_mixed_array_dtypes(first_dtype, second_dtype):
    first = gs.array([0.0, 1.0, 1.0 + 1e-7], dtype=first_dtype)
    second = gs.array([[0.0], [1.0 + 8e-8]], dtype=second_dtype)
    expected = np.isclose(gs.to_numpy(first), gs.to_numpy(second), rtol=0.0, atol=1e-9)

    result = gs.isclose(first, second, rtol=0.0, atol=1e-9)

    np.testing.assert_array_equal(gs.to_numpy(result), expected)
    assert gs.allclose(first, second, rtol=0.0, atol=1e-9) == np.all(expected)
    assert gs.allclose(
        gs.array([0, 1, 2], dtype=first_dtype),
        gs.array([0, 1, 2], dtype=second_dtype),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "dtype", [gs.int32, gs.int64, gs.float32, gs.float64, gs.complex64, gs.complex128]
)
@pytest.mark.parametrize("scalar", [0, 0.0, 0j])
@pytest.mark.parametrize("reverse", [False, True])
def test_close_array_and_scalar(dtype, scalar, reverse):
    array = gs.array([0, 1, 0], dtype=dtype)
    first, second = (scalar, array) if reverse else (array, scalar)

    result = gs.isclose(first, second)

    np.testing.assert_array_equal(gs.to_numpy(result), [True, False, True])
    assert not gs.allclose(first, second)
    assert gs.allclose(gs.array([0, 0], dtype=dtype), scalar)


def test_online_kmeans_float32_scalar_convergence(monkeypatch):
    data = gs.array([[0.0, 0.0], [1.0, 1.0]], dtype=gs.float32)
    indices = iter([gs.array([0]), gs.array([1])])
    monkeypatch.setattr(gs.random, "randint", lambda **kwargs: next(indices))

    estimator = OnlineKMeans(Euclidean(2), n_clusters=1, max_iter=1).fit(data)

    np.testing.assert_allclose(gs.to_numpy(estimator.cluster_centers_), [[0.5, 0.5]])
    assert estimator.cluster_centers_.dtype == gs.float32
