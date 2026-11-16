import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.numerics.ivp import GSIVPIntegrator, ScipySolveIVP


@pytest.mark.parametrize("point_shape", [(2,), (2, 2)])
@pytest.mark.parametrize("n_batch", [1, 2, 3])
@pytest.mark.parametrize("n_times", [1, 3])
def test_chosen_time_batch_last_state(point_shape, n_batch, n_times):
    shape = (n_batch, 2) + point_shape
    initial = gs.reshape(gs.arange(float(np.prod(shape))), shape)
    times = gs.array([1.0]) if n_times == 1 else gs.linspace(0.0, 1.0, n_times)
    integrator = ScipySolveIVP(point_ndim=len(point_shape), save_result=True)

    result = integrator.integrate_t(
        lambda state, time: 0.5 * gs.ones_like(state), initial, times
    )

    expected_path = initial[:, None] + 0.5 * gs.reshape(
        times, (1, n_times) + (1,) * (len(point_shape) + 1)
    )
    assert integrator.result_ is result
    assert result.y.shape == (n_batch, n_times, 2) + point_shape
    np.testing.assert_allclose(gs.to_numpy(result.y), gs.to_numpy(expected_path))
    assert result.get_last_y().shape == initial.shape
    np.testing.assert_allclose(
        gs.to_numpy(result.get_last_y()), gs.to_numpy(initial + 0.5)
    )


@pytest.mark.parametrize("point_shape", [(2,), (2, 2)])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("kind", ["adaptive", "fixed_step"])
def test_last_state_other_integrators(point_shape, batched, kind):
    shape = ((3,) if batched else ()) + (2,) + point_shape
    initial = gs.reshape(gs.arange(float(np.prod(shape))), shape)
    integrator = (
        ScipySolveIVP(point_ndim=len(point_shape))
        if kind == "adaptive"
        else GSIVPIntegrator(n_steps=4, step_type="rk4")
    )

    result = integrator.integrate(
        lambda state, time: 0.5 * gs.ones_like(state), initial
    )

    assert result.get_last_y().shape == initial.shape
    np.testing.assert_allclose(
        gs.to_numpy(result.get_last_y()), gs.to_numpy(initial + 0.5)
    )


@pytest.mark.parametrize("point_shape", [(2,), (2, 2)])
def test_chosen_time_unbatched_last_state(point_shape):
    shape = (2,) + point_shape
    initial = gs.reshape(gs.arange(float(np.prod(shape))), shape)
    integrator = ScipySolveIVP(point_ndim=len(point_shape))

    result = integrator.integrate_t(
        lambda state, time: 0.5 * gs.ones_like(state),
        initial,
        gs.array([0.0, 0.25, 1.0]),
    )

    assert result.get_last_y().shape == initial.shape
    np.testing.assert_allclose(
        gs.to_numpy(result.get_last_y()), gs.to_numpy(initial + 0.5)
    )
