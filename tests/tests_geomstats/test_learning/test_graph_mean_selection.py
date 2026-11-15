"""Regressions for graph Frechet mean automatic algorithm selection."""

import itertools

import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.metric_geometry.graph_space import GraphSpace


def _graphs(n_nodes):
    graph = np.diag(10.0 * 2.0 ** np.arange(n_nodes))
    graph[0, 1] = 0.5
    graph[1, 2] = 1.5
    graph[2, 0] = 0.8
    if n_nodes == 3:
        graph = graph + graph.T
    permutations = [
        np.arange(n_nodes),
        np.roll(np.arange(n_nodes), 1),
        np.arange(n_nodes)[::-1],
    ]
    samples = np.stack(
        [
            scale * graph[np.ix_(permutation, permutation)]
            for scale, permutation in zip([0.6, 1.0, 1.4], permutations)
        ]
    )
    return gs.array(samples), graph


def _graph_space(n_nodes, align):
    space = GraphSpace(n_nodes)
    if align:
        space.equip_with_group_action()
        space.equip_with_quotient()
        space.aligner.set_alignment_algorithm("exhaustive")
    return space


def _assert_same_unlabeled_graph(actual, expected):
    actual = gs.to_numpy(actual)
    errors = [
        np.linalg.norm(actual[np.ix_(permutation, permutation)] - expected)
        for permutation in itertools.permutations(range(len(expected)))
    ]
    assert min(errors) < 1e-9


@pytest.mark.parametrize("n_nodes", [3, 4])
@pytest.mark.parametrize("method", [None, "auto", "aac"])
def test_graph_mean_aligns_permuted_observations(n_nodes, method):
    samples, expected = _graphs(n_nodes)
    original = gs.copy(samples)
    space = _graph_space(n_nodes, align=True)
    kwargs = {} if method is None else {"method": method}
    gs.random.seed(7)
    estimator = FrechetMean(space, **kwargs)

    assert estimator.fit(samples) is estimator

    _assert_same_unlabeled_graph(estimator.estimate_, expected)
    np.testing.assert_array_equal(gs.to_numpy(samples), gs.to_numpy(original))


@pytest.mark.parametrize("n_nodes", [3, 4])
@pytest.mark.parametrize("align", [False, True])
def test_labeled_graph_mean_keeps_arithmetic_mean(n_nodes, align):
    samples, _ = _graphs(n_nodes)
    space = _graph_space(n_nodes, align=align)
    kwargs = {"method": "linear"} if align else {}

    estimator = FrechetMean(space, **kwargs).fit(samples)

    np.testing.assert_allclose(
        gs.to_numpy(estimator.estimate_), gs.to_numpy(samples).mean(axis=0)
    )


@pytest.mark.parametrize("n_nodes", [3, 4])
def test_automatic_graph_mean_accepts_alignment_estimator_options(n_nodes):
    samples, expected = _graphs(n_nodes)
    estimator = FrechetMean(
        _graph_space(n_nodes, align=True),
        init_point=gs.array(expected),
        save_last_X=False,
        epsilon=1e-10,
    ).fit(samples)

    _assert_same_unlabeled_graph(estimator.estimate_, expected)
    assert estimator.aligned_X_ is None
