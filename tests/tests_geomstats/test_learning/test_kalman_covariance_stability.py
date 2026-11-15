"""Numerical regressions for the Kalman posterior covariance update."""

import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.learning.kalman_filter import (
    KalmanFilter,
    Localization,
    LocalizationLinear,
)


@pytest.mark.parametrize("dtype,prior_scale", [(gs.float32, 1e12), (gs.float64, 1e20)])
def test_high_dynamic_range_filter_continues_using_new_measurements(dtype, prior_scale):
    model = LocalizationLinear()
    estimator = KalmanFilter(model)
    prior = gs.array([[prior_scale, 0.0], [0.0, 2.0]], dtype=dtype)
    noise = gs.array([[1.0]], dtype=dtype)
    estimator.initialize_covariances(prior, gs.zeros((1, 1)), noise)
    original_prior = gs.copy(prior)
    original_noise = gs.copy(noise)

    estimator.update(gs.array([1.0], dtype=dtype))
    first_covariance = gs.to_numpy(estimator.covariance).copy()
    estimator.update(gs.array([3.0], dtype=dtype))

    expected_variance = 1 / (1 / prior_scale + 2)
    expected_position = expected_variance * (1.0 + 3.0)
    np.testing.assert_allclose(
        gs.to_numpy(estimator.state), [expected_position, 0.0], rtol=1e-6
    )
    np.testing.assert_allclose(
        first_covariance, np.diag([1 / (1 / prior_scale + 1), 2.0]), rtol=1e-6
    )
    np.testing.assert_allclose(
        gs.to_numpy(estimator.covariance), np.diag([expected_variance, 2.0]), rtol=1e-6
    )
    np.testing.assert_array_equal(gs.to_numpy(prior), gs.to_numpy(original_prior))
    np.testing.assert_array_equal(gs.to_numpy(noise), gs.to_numpy(original_noise))


@pytest.mark.parametrize("kind", ["linear", "invariant"])
@pytest.mark.parametrize("prior_scale", [1.0, 1e20])
def test_covariance_matches_independent_information_filter(kind, prior_scale):
    if kind == "linear":
        model = LocalizationLinear()
        state = np.zeros(2)
        prior = np.diag([prior_scale, 2.0])
        noise = np.array([[0.7]])
        transformed_noise = noise
        observation_matrix = np.array([[1.0, 0.0]])
        observation = np.zeros(1)
    else:
        model = Localization()
        angle = 0.7
        state = np.array([angle, 0.3, -0.2])
        prior = np.diag([0.4, prior_scale, 2 * prior_scale])
        noise = np.array([[0.7, 0.2], [0.2, 1.5]])
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        transformed_noise = rotation.T @ noise @ rotation
        observation_matrix = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        observation = state[1:]
    expected = np.linalg.inv(
        np.linalg.inv(prior)
        + observation_matrix.T @ np.linalg.solve(transformed_noise, observation_matrix)
    )
    estimator = KalmanFilter(model)
    estimator.state = gs.array(state)
    estimator.initialize_covariances(
        gs.array(prior), gs.zeros((model.dim_noise, model.dim_noise)), gs.array(noise)
    )

    estimator.update(gs.array(observation))

    actual = gs.to_numpy(estimator.covariance)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(actual, actual.T, atol=1e-12)
    assert np.linalg.eigvalsh(actual).min() > 0
    np.testing.assert_allclose(gs.to_numpy(estimator.state), state, atol=1e-12)
