import numpy as np
import pytest
from scipy.stats import beta as scipy_beta

import geomstats.backend as gs
from geomstats.information_geometry.beta import (
    BetaDistributions,
    BetaDistributionsRandomVariable,
    BetaMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.information_geometry.beta import (
    BetaDistributionsTestCase,
    BetaMetricTestCase,
)

from .data.beta import (
    BetaDistributionsSmokeTestData,
    BetaDistributionsTestData,
    BetaMetricSmokeTestData,
    BetaMetricTestData,
)


class TestBetaDistributions(BetaDistributionsTestCase, metaclass=DataBasedParametrizer):
    space = BetaDistributions(equip=False)
    random_variable = BetaDistributionsRandomVariable(space)
    testing_data = BetaDistributionsTestData()


@pytest.mark.smoke
class TestBetaDistributionsSmoke(
    BetaDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = BetaDistributions(equip=False)
    testing_data = BetaDistributionsSmokeTestData()


@pytest.mark.slow
class TestBetaMetric(BetaMetricTestCase, metaclass=DataBasedParametrizer):
    space = BetaDistributions()

    data_generator = RandomDataGenerator(space, amplitude=5.0)

    testing_data = BetaMetricTestData()


@pytest.mark.smoke
class TestBetaMetricSmoke(BetaMetricTestCase, metaclass=DataBasedParametrizer):
    space = BetaDistributions(equip=False)
    space.equip_with_metric(BetaMetric)
    testing_data = BetaMetricSmokeTestData()


@pytest.mark.parametrize("loc, scale", [(0.0, 1.0), (10.0, 2.0), (-3.0, 0.5)])
@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
@pytest.mark.parametrize("container", ["array", "list", "tuple"])
def test_beta_fit_sample_batches(loc, scale, batch_shape, container):
    count = int(np.prod(batch_shape))
    samples = np.stack(
        [np.linspace(0.0, 1.0, 9) ** (1 + 0.1 * i) for i in range(count)]
    )
    epsilon = 1e-4
    expected = np.array(
        [
            scipy_beta.fit(np.clip(sample, epsilon, 1 - epsilon), floc=0, fscale=1)[:2]
            for sample in samples
        ]
    ).reshape(batch_shape + (2,))
    original = (loc + scale * samples).reshape(batch_shape + (9,))
    data = gs.array(original) if container == "array" else original.tolist()
    if container == "tuple":
        data = tuple(data)

    result = BetaDistributions.maximum_likelihood_fit(
        data, loc=loc, scale=scale, epsilon=epsilon
    )

    assert result.shape == batch_shape + (2,)
    np.testing.assert_allclose(gs.to_numpy(result), expected, rtol=1e-8, atol=1e-9)
    unchanged = gs.to_numpy(data) if gs.is_array(data) else np.array(data)
    np.testing.assert_array_equal(unchanged, original)


@pytest.mark.parametrize("container", ["list", "tuple", "backend_arrays"])
def test_beta_fit_ragged_samples(container):
    samples = [np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 13) ** 2]
    data = [(10 + 2 * sample).tolist() for sample in samples]
    if container == "tuple":
        data = tuple(tuple(sample) for sample in data)
    elif container == "backend_arrays":
        data = [gs.array(sample) for sample in data]
    expected = [
        scipy_beta.fit(np.clip(sample, 1e-4, 1 - 1e-4), floc=0, fscale=1)[:2]
        for sample in samples
    ]

    result = BetaDistributions.maximum_likelihood_fit(
        data, loc=10.0, scale=2.0, epsilon=1e-4
    )

    assert result.shape == (2, 2)
    np.testing.assert_allclose(gs.to_numpy(result), expected, rtol=1e-8, atol=1e-9)


def test_beta_fit_rejects_samples_outside_support():
    with pytest.raises(ValueError):
        BetaDistributions.maximum_likelihood_fit(gs.array([-0.1, 0.5, 1.1]))
