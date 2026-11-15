"""Regressions for singleton component axes in Riemannian EM."""

import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.expectation_maximization import (
    GaussianMixtureModel,
    RiemannianEM,
)


def _data(times):
    times = gs.array(times)
    return gs.stack((gs.tanh(times / 2), gs.zeros_like(times)), axis=-1)


@pytest.mark.parametrize("initialisation", ["random", "kmeans"])
@pytest.mark.parametrize("times", [[0.4], [-0.7, -0.2, 0.3, 0.8]])
def test_single_component_fit_matches_diameter_mean(initialisation, times):
    gs.random.seed(7)
    space = PoincareBall(2)
    samples = _data(times)
    estimator = RiemannianEM(
        space,
        n_gaussians=1,
        initialisation_method=initialisation,
        max_iter=20,
    )

    assert estimator.fit(samples) is estimator

    mean_time = np.mean(times)
    expected_mean = [np.tanh(mean_time / 2), 0.0]
    assert estimator.means_.shape == (1, 2)
    assert estimator.variances_.shape == (1,)
    assert estimator.mixture_coefficients_.shape == (1,)
    np.testing.assert_allclose(
        gs.to_numpy(estimator.means_[0]), expected_mean, atol=1e-6
    )
    np.testing.assert_allclose(
        gs.to_numpy(estimator.mixture_coefficients_), [1.0], atol=1e-12
    )

    # Hyperbolic distance along this diameter is the difference of the times.
    scale = float(gs.to_numpy(estimator.variances_[0]))
    normalizer = float(
        gs.to_numpy(space.metric.normalization_factor(gs.array([scale])))[0]
    )
    expected_density = np.exp(
        -np.square(np.asarray(times) - mean_time) / (2 * scale**2)
    ) / normalizer
    density = estimator._model.pdf(samples)
    assert density.shape == (len(times), 1)
    np.testing.assert_allclose(
        gs.to_numpy(density[:, 0]), expected_density, rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("initialisation", ["random", "kmeans"])
def test_refit_preserves_component_axes(initialisation):
    gs.random.seed(11)
    samples = _data([-0.9, -0.6, -0.2, 0.1, 0.5, 0.8])
    estimator = RiemannianEM(
        PoincareBall(2),
        n_gaussians=2,
        initialisation_method=initialisation,
        max_iter=20,
    )

    for count in (2, 1, 2):
        estimator.set_params(n_gaussians=count)
        assert estimator.fit(samples) is estimator
        assert estimator.means_.shape == (count, 2)
        assert estimator.variances_.shape == (count,)
        assert estimator.mixture_coefficients_.shape == (count,)
        density = gs.to_numpy(estimator._model.pdf(samples))
        assert density.shape == (len(samples), count)
        assert np.isfinite(density).all()
        assert (density > 0).all()
        assert gs.all(estimator.space.belongs(estimator.means_))


@pytest.mark.parametrize("initialisation", ["random", "kmeans"])
def test_multiple_components_retain_existing_shape(initialisation):
    gs.random.seed(5)
    samples = _data([-0.8, -0.5, -0.1, 0.2, 0.6, 0.9])
    estimator = RiemannianEM(
        PoincareBall(2),
        n_gaussians=2,
        initialisation_method=initialisation,
        max_iter=20,
    ).fit(samples)

    assert estimator.means_.shape == (2, 2)
    assert estimator.variances_.shape == (2,)
    assert estimator._model.pdf(samples).shape == (6, 2)
    np.testing.assert_allclose(
        gs.to_numpy(estimator.mixture_coefficients_).sum(), 1.0, atol=1e-8
    )


@pytest.mark.parametrize("count", [1, 2, 3])
def test_densities_include_every_sample_component_pair(count):
    times = np.array([-0.3, 0.4, 0.8])[:count]
    centers = np.array([-0.2, 0.3, 0.6])[:count]
    space = PoincareBall(2)
    scale = 0.7
    model = GaussianMixtureModel(
        space, means=_data(centers), variances=gs.ones(count) * scale
    )
    samples = _data(times)
    squared_distances = (times[:, None] - centers[None, :]) ** 2
    reference = np.exp(-squared_distances / (2 * scale**2))
    normalizer = gs.to_numpy(
        space.metric.normalization_factor(gs.array([scale]))
    )[0]

    density = gs.to_numpy(model.pdf(samples))
    assert density.shape == (count, count)
    np.testing.assert_allclose(density, reference / normalizer, rtol=1e-6)

    weighted = gs.to_numpy(model.weighted_pdf(gs.ones(count) / count, samples))
    assert weighted.shape == (count, count)
    # Equal scales share a normalizer, so row ratios depend only on distances.
    np.testing.assert_allclose(
        weighted / weighted.sum(axis=1, keepdims=True),
        reference / reference.sum(axis=1, keepdims=True),
        rtol=1e-6,
    )


@pytest.mark.parametrize("initialisation", ["random", "kmeans"])
def test_fit_with_equal_sample_and_component_counts(initialisation):
    gs.random.seed(5)
    samples = _data([-0.5, 0.6])
    estimator = RiemannianEM(
        PoincareBall(2),
        n_gaussians=2,
        initialisation_method=initialisation,
        max_iter=20,
    ).fit(samples)

    assert estimator.means_.shape == (2, 2)
    assert estimator.variances_.shape == (2,)
    assert estimator._model.pdf(samples).shape == (2, 2)
    assert np.isfinite(gs.to_numpy(estimator.variances_)).all()
