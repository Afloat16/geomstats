import threading

import joblib
import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.frechet_mean import LinearMean
from geomstats.learning.riemannian_mean_shift import RiemannianMeanShift


@pytest.mark.parametrize("n_jobs", [2, 4])
def test_mean_shift_threaded_center_updates(n_jobs, monkeypatch):
    barrier = threading.Barrier(2, timeout=10)
    configured_calls = []

    class SynchronizedMean(LinearMean):
        def fit(self, X, y=None, weights=None):
            super().fit(X, y, weights)
            configured_calls.append(self.marker)
            barrier.wait()
            return self

    space = Euclidean(2)
    data = gs.array([[0.0, 0.0], [0.2, 0.0], [3.0, 3.0], [3.2, 3.0]])
    estimator = RiemannianMeanShift(
        space, bandwidth=0.5, n_clusters=2, n_jobs=n_jobs, max_iter=1
    )
    estimator.mean_estimator = SynchronizedMean(space)
    estimator.mean_estimator.marker = "configured"
    monkeypatch.setattr(estimator, "_initialization", lambda X: data[[0, 2]])

    with joblib.parallel_backend("threading"):
        estimator.fit(data)

    np.testing.assert_allclose(
        gs.to_numpy(estimator.cluster_centers_), [[0.1, 0.0], [3.1, 3.0]]
    )
    np.testing.assert_array_equal(gs.to_numpy(estimator.predict(data)), [0, 0, 1, 1])
    assert configured_calls == ["configured", "configured"]
