import numpy as np
import pytest
from scipy import special, stats

import geomstats.backend as gs
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.information_geometry.gamma import GammaDistributions
from geomstats.test.test_case import autodiff_only


@pytest.mark.parametrize("family", ["beta", "gamma"])
@pytest.mark.parametrize("dtype", [gs.float32, gs.float64])
def test_density_concentration_and_support(family, dtype):
    if family == "beta":
        space = BetaDistributions(equip=False)
        points = [[0.5, 0.5], [1, 2], [2, 1], [2, 3], [100, 100], [1000, 500]]
    else:
        space = GammaDistributions(equip=False)
        points = [[0.5, 1], [1, 2], [2, 3], [100, 1], [200, 2], [1000, 5]]
    points = gs.reshape(gs.array(points, dtype=dtype), (2, 3, 2))
    samples = gs.array([-1, 0, 0.2, 0.5, 0.8, 1, 2, 3, 5, np.nan], dtype=dtype)
    first = gs.to_numpy(points[..., 0, None])
    second = gs.to_numpy(points[..., 1, None])
    if family == "beta":
        expected = stats.beta.pdf(gs.to_numpy(samples), first, second)
    else:
        expected = stats.gamma.pdf(gs.to_numpy(samples), first, scale=second / first)

    result = space.point_to_pdf(points)(samples)

    assert result.shape == (2, 3, len(samples))
    rtol, atol = (2e-3, 1e-7) if dtype == gs.float32 else (1e-10, 1e-14)
    np.testing.assert_allclose(gs.to_numpy(result), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "space", [BetaDistributions(equip=False), GammaDistributions(equip=False)]
)
def test_density_at_infinity(space):
    result = space.point_to_pdf(gs.array([200.0, 3.0]))(gs.array([-np.inf, np.inf]))
    np.testing.assert_array_equal(gs.to_numpy(result), [0.0, 0.0])


@pytest.mark.parametrize("family", ["beta", "gamma"])
@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
def test_density_concentrated_batch_shape(family, batch_shape):
    if family == "beta":
        space = BetaDistributions(equip=False)
        point = gs.array([100.0, 120.0])
        samples = gs.array([0.4, 0.45, 0.5])
        expected = stats.beta.pdf(gs.to_numpy(samples), 100, 120)
    else:
        space = GammaDistributions(equip=False)
        point = gs.array([200.0, 3.0])
        samples = gs.array([2.8, 3.0, 3.2])
        expected = stats.gamma.pdf(gs.to_numpy(samples), 200, scale=3 / 200)
    points = gs.broadcast_to(point, batch_shape + (2,))

    result = space.point_to_pdf(points)(samples)

    assert result.shape == batch_shape + (3,)
    np.testing.assert_allclose(
        gs.to_numpy(result), np.broadcast_to(expected, result.shape), rtol=1e-10
    )


@autodiff_only
@pytest.mark.parametrize(
    "family, point, samples",
    [
        ("beta", [100.0, 120.0], [0.4, 0.45, 0.5]),
        ("beta", [1.0, 2.0], [0.2, 0.4, 0.6]),
        ("gamma", [200.0, 3.0], [2.8, 3.0, 3.2]),
        ("gamma", [1.0, 2.0], [0.5, 1.0, 1.5]),
    ],
)
def test_pdf_parameter_gradient(family, point, samples):
    first, second = point
    samples_np = np.array(samples)
    if family == "beta":
        space = BetaDistributions(equip=False)
        density = stats.beta.pdf(samples_np, first, second)
        log_gradient = np.stack(
            [
                np.log(samples_np)
                - special.digamma(first)
                + special.digamma(sum(point)),
                np.log1p(-samples_np)
                - special.digamma(second)
                + special.digamma(sum(point)),
            ],
            axis=-1,
        )
    else:
        space = GammaDistributions(equip=False)
        density = stats.gamma.pdf(samples_np, first, scale=second / first)
        log_gradient = np.stack(
            [
                np.log(first)
                + 1
                - np.log(second)
                - special.digamma(first)
                + np.log(samples_np)
                - samples_np / second,
                -first / second + first * samples_np / second**2,
            ],
            axis=-1,
        )

    result = gs.autodiff.jacobian(
        lambda parameter: space.point_to_pdf(parameter)(gs.array(samples))
    )(gs.array(point))

    np.testing.assert_allclose(
        gs.to_numpy(result), density[..., None] * log_gradient, rtol=1e-8, atol=1e-10
    )


def test_gammaln_against_scipy():
    values = gs.array([-2.5, -0.5, 0.1, 1.0, 10.0, 1000.0])
    np.testing.assert_allclose(
        gs.to_numpy(gs.gammaln(values)), special.gammaln(gs.to_numpy(values))
    )


@autodiff_only
def test_gammaln_gradient():
    values = gs.array([0.1, 1.0, 10.0, 1000.0])
    _, gradient = gs.autodiff.value_and_grad(lambda point: gs.sum(gs.gammaln(point)))(
        values
    )
    np.testing.assert_allclose(
        gs.to_numpy(gradient), special.digamma(gs.to_numpy(values))
    )
