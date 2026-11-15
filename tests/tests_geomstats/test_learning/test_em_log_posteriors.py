"""Numerical regressions for the Riemannian EM expectation step."""

import numpy as np
import pytest
from scipy.special import softmax

import geomstats.backend as gs
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.expectation_maximization import (
    GaussianMixtureModel,
    RiemannianEM,
)


def _diameter(times):
    times = gs.array(times)
    return gs.stack((gs.tanh(times / 2), gs.zeros_like(times)), axis=-1)


@pytest.mark.parametrize(
    "times,centers,scales,priors",
    [
        ([-8.0, -7.0, 7.0, 8.0], [-0.2, 0.3], [0.5, 0.5], [0.25, 0.75]),
        ([-0.8, -0.7, 0.7, 0.8], [-0.002, 0.003], [0.01, 0.01], [0.25, 0.75]),
        ([-7.0, 6.5, 7.3], [-0.2, 0.4], [0.5, 0.8], [0.3, 0.7]),
        ([-0.7, 0.1, 0.6, 0.9], [-0.2, 0.3], [0.5, 0.8], [0.3, 0.7]),
    ],
    ids=["low-density", "underflow", "unequal-scales", "ordinary-density"],
)
def test_expectation_preserves_bayes_ratios(times, centers, scales, priors):
    space = PoincareBall(2)
    estimator = RiemannianEM(space, n_gaussians=2)
    estimator._model = GaussianMixtureModel(
        space,
        means=_diameter(centers),
        variances=gs.array(scales),
        zeta_lower_bound=0.01,
        zeta_upper_bound=2.0,
        zeta_step=0.01,
    )
    estimator.mixture_coefficients_ = gs.array(priors)
    samples = _diameter(times)

    # Along a Poincare diameter the arclengths are the supplied times.
    distances = np.asarray(times)[:, None] - np.asarray(centers)[None, :]
    normalization = gs.to_numpy(
        space.metric.normalization_factor(gs.array(scales))
    )
    logits = (
        -distances**2 / (2 * np.square(scales))
        - np.log(normalization)
        + np.log(priors)
    )
    expected = softmax(logits, axis=1)

    actual = gs.to_numpy(estimator._expectation(samples))

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-12)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, atol=1e-12)
    assert np.isfinite(actual).all()
    if min(scales) == 0.01:
        # This exercises actual floating-point underflow in the public PDF.
        np.testing.assert_array_equal(gs.to_numpy(estimator._model.pdf(samples)), 0)


def test_public_fit_retains_information_from_low_density_samples():
    space = PoincareBall(2)
    centers = [-0.2, 0.2]
    times = [-8.0, -7.0, 7.0]
    samples = _diameter(times)

    class InitializedEM(RiemannianEM):
        def _initialization(self, X):
            return _diameter(centers), gs.array([0.5, 0.5])

    estimator = InitializedEM(space, n_gaussians=2, max_iter=1)
    distances = np.asarray(times)[:, None] - np.asarray(centers)[None, :]
    expected = softmax(-distances**2 / (2 * 0.5**2), axis=1).mean(axis=0)

    assert estimator.fit(samples) is estimator

    np.testing.assert_allclose(
        gs.to_numpy(estimator.mixture_coefficients_), expected, rtol=2e-5
    )
    assert float(estimator.means_[0, 0]) < 0 < float(estimator.means_[1, 0])
    assert gs.all(gs.isfinite(estimator.means_))
    assert gs.all(space.belongs(estimator.means_))


@pytest.mark.parametrize("priors", [[0.0, 1.0], [1.0, 0.0]])
def test_empty_component_regularization_keeps_normalized_posteriors(priors):
    estimator = RiemannianEM(PoincareBall(2), n_gaussians=2)
    estimator._model.means = _diameter([-0.2, 0.3])
    estimator._model.variances = gs.array([0.5, 0.5])
    estimator.mixture_coefficients_ = gs.array(priors)

    probabilities = estimator._expectation(_diameter([-0.1, 0.0, 0.2]))
    values = gs.to_numpy(probabilities)

    np.testing.assert_allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert (values > 0).all()
    estimator._update_posterior_probabilities(probabilities)
    np.testing.assert_allclose(
        gs.to_numpy(estimator.mixture_coefficients_).sum(), 1.0, rtol=0, atol=1e-12
    )
