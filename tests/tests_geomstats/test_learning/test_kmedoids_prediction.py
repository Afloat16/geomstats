import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.kmedoids import RiemannianKMedoids
from geomstats.test.test_case import torch_only


@torch_only
def test_predict_does_not_round_double_distances_to_the_default_dtype():
    """The nearest medoid must be chosen before converting the label dtype."""
    previous_dtype = str(gs.get_default_dtype()).split(".")[-1]
    try:
        gs.set_default_dtype("float32")
        estimator = RiemannianKMedoids(Euclidean(1), n_clusters=2)
        estimator.cluster_centers_ = gs.array([[1.0 + 1e-8], [1.0]], dtype=gs.float64)

        prediction = estimator.predict(gs.array([[0.0]], dtype=gs.float64))

        assert prediction.shape == (1,)
        assert prediction[0] == 1
        assert prediction.dtype == gs.get_default_dtype()
    finally:
        gs.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("matrix_points", [False, True])
@pytest.mark.parametrize("n_samples", [0, 1, 16])
def test_predict_matches_individual_metric_distances(matrix_points, n_samples):
    if matrix_points:
        space = SPDMatrices(2)
        centers = gs.array([[[1.0, 0.0], [0.0, 1.0]], [[4.0, 0.0], [0.0, 9.0]]])
        scales = gs.linspace(1.0, 4.0, 16)
        samples = scales[:, None, None] * gs.eye(2)
    else:
        space = Euclidean(2)
        centers = gs.array([[0.0, 0.0], [3.0, 2.0]])
        samples = gs.reshape(gs.linspace(0.0, 5.0, 32), (16, 2))
    samples = samples[:n_samples]
    estimator = RiemannianKMedoids(space, n_clusters=2)
    estimator.cluster_centers_ = centers
    expected = gs.array(
        [
            gs.argmin(
                gs.stack([space.metric.dist(point, center) for center in centers])
            )
            for point in samples
        ]
    )

    prediction = estimator.predict(samples)

    assert prediction.shape == (n_samples,)
    assert gs.all(prediction == expected)
    assert prediction.dtype == gs.get_default_dtype()
