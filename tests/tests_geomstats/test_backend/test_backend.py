import pytest

import geomstats.backend as gs
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.backend import BackendTestCase

from .data.backend import BackendTestData


@pytest.mark.parametrize("weighted", [False, True])
def test_random_choice_without_replacement(weighted):
    """Sampling the complete population must return a permutation."""
    population = gs.arange(7)
    probabilities = gs.array([1.0 / 7] * 7) if weighted else None

    sample = gs.random.choice(population, 7, p=probabilities, replace=False)

    assert gs.all(gs.sort(sample) == population)


@pytest.mark.parametrize("sample_size", [-1, 4])
def test_random_choice_without_replacement_invalid_size(sample_size):
    with pytest.raises(ValueError):
        gs.random.choice(gs.arange(3), sample_size, replace=False)


@pytest.mark.parametrize("weighted", [False, True])
def test_random_choice_keeps_replacement_by_default(weighted):
    population = gs.array([7])
    probabilities = gs.array([1.0]) if weighted else None

    sample = gs.random.choice(population, 4, p=probabilities)

    assert gs.all(sample == gs.array([7, 7, 7, 7]))


def test_where_without_values():
    """Check that where without values returns matching indices."""
    (indices,) = gs.where([False, True, False, True])

    assert gs.all(indices == gs.array([1, 3]))


class TestBackend(BackendTestCase, metaclass=DataBasedParametrizer):
    testing_data = BackendTestData()
