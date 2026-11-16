import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_validate

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean, GeneralFrechetMean


@pytest.mark.parametrize("optimizer_kwargs", [None, {}, {"max_iter": 64}])
@pytest.mark.parametrize("method", ["default", "adaptive"])
@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("factory", [FrechetMean, GeneralFrechetMean])
def test_clone_frechet_mean(optimizer_kwargs, method, fitted, factory):
    space = Hypersphere(2)
    point = gs.array([1.0, 0.0, 0.0])
    data = space.metric.exp(gs.array([[0.0, -0.3, 0.0], [0.0, 0.3, 0.0]]), point)
    estimator = factory(space, method=method, optimizer_kwargs=optimizer_kwargs)
    if fitted:
        estimator.fit(data)
    old_estimate = estimator.estimate_
    old_optimizer = estimator.optimizer_

    copied = clone(estimator)

    assert copied is not estimator
    assert copied.method == method
    assert copied.optimizer_kwargs == estimator.optimizer_kwargs
    assert copied.optimizer_kwargs is not estimator.optimizer_kwargs
    assert copied.estimate_ is None
    assert copied.optimizer_ is None
    copied.fit(data)
    np.testing.assert_allclose(gs.to_numpy(copied.estimate_), gs.to_numpy(point))
    assert estimator.estimate_ is old_estimate
    assert estimator.optimizer_ is old_optimizer
    copied.optimizer_kwargs["max_iter"] = 1
    assert estimator.optimizer_kwargs.get("max_iter") != 1


def test_frechet_mean_cross_validation():
    space = Hypersphere(2)
    point = gs.array([1.0, 0.0, 0.0])
    data = space.metric.exp(
        gs.array(
            [[0.0, -0.3, 0.0], [0.0, 0.3, 0.0], [0.0, -0.2, 0.0], [0.0, 0.2, 0.0]]
        ),
        point,
    )

    def score(estimator, samples):
        distances = estimator.space.metric.squared_dist(samples, estimator.estimate_)
        return -float(gs.to_numpy(gs.mean(distances)))

    result = cross_validate(
        FrechetMean(space), data, scoring=score, cv=2, error_score="raise"
    )

    np.testing.assert_allclose(result["test_score"], [-0.09, -0.04], atol=1e-10)
