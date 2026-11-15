"""Regression tests for selecting an existing graph aligner instance."""

import itertools

import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.metric_geometry.graph_space import ExhaustiveAligner, GraphSpace


def _space(use_instance):
    space = GraphSpace(3)
    space.equip_with_group_action()
    space.equip_with_quotient()
    algorithm = ExhaustiveAligner(space) if use_instance else "exhaustive"
    selected = space.aligner.set_alignment_algorithm(algorithm)
    if use_instance:
        assert selected is algorithm
    return space


def _graphs():
    target = np.array([[2.0, 0.5, 0.3], [0.7, 5.0, 0.8], [0.2, 0.6, 11.0]])
    permutations = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
    points = np.stack(
        [target[np.ix_(permutation, permutation)] for permutation in permutations]
    )
    return target, points


def _distance(point, target):
    return min(
        np.linalg.norm(point[np.ix_(permutation, permutation)] - target)
        for permutation in itertools.permutations(range(3))
    )


@pytest.mark.parametrize("use_instance", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_selected_aligner_computes_public_alignment_and_distance(use_instance, batched):
    space = _space(use_instance)
    target, points = _graphs()
    points = points * np.array([0.7, 1.0, 1.3])[:, None, None]
    if not batched:
        points = points[2]
    original = points.copy()

    aligned = space.aligner.align(gs.array(points), gs.array(target))
    actual = space.quotient.metric.dist(gs.array(points), gs.array(target))

    reference_points = points if batched else points[None]
    expected = np.array([_distance(point, target) for point in reference_points])
    if not batched:
        expected = expected[0]
    np.testing.assert_allclose(gs.to_numpy(actual), expected, atol=1e-10)
    np.testing.assert_allclose(
        np.linalg.norm(gs.to_numpy(aligned) - target, axis=(-2, -1)),
        expected,
        atol=1e-10,
    )
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize("use_instance", [False, True])
def test_selected_aligner_is_used_by_real_graph_mean(use_instance):
    space = _space(use_instance)
    target, points = _graphs()
    estimator = FrechetMean(
        space, method="aac", init_point=gs.array(target), epsilon=1e-10
    ).fit(gs.array(points))

    assert _distance(gs.to_numpy(estimator.estimate_), target) < 1e-10
    np.testing.assert_allclose(
        gs.to_numpy(estimator.aligned_X_), np.broadcast_to(target, points.shape)
    )
