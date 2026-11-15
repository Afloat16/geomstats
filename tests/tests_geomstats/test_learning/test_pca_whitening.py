"""Whitening through public TangentPCA estimator methods."""

import numpy as np
import pytest
from sklearn.decomposition import PCA

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.pca import TangentPCA
from geomstats.test.test_case import np_and_autograd_only

SAMPLES = np.array(
    [
        [2.0, -1.0, 0.2],
        [0.0, 1.5, -0.4],
        [-3.0, 0.5, 0.7],
        [4.0, 2.0, -1.0],
        [1.0, -2.0, 0.4],
        [-2.0, -0.5, -1.0],
        [0.3, 2.0, 2.0],
        [2.5, 1.5, -0.7],
    ]
)
QUERIES = np.array([[0.5, -0.2, 0.8], [-1.5, 0.4, -0.6]])


def _aligned_reference(model, reference, scores):
    signs = np.sign(np.sum(model.components_ * reference.components_, axis=1))
    return scores * signs


@np_and_autograd_only
class TestTangentPCAWhitening:
    @pytest.mark.parametrize("n_components", [1, 2, 3])
    @pytest.mark.parametrize("entry", ["fit", "fit_transform"])
    def test_whitening_matches_euclidean_pca(self, n_components, entry):
        data = gs.array(SAMPLES)
        model = TangentPCA(Euclidean(3), n_components=n_components, whiten=True)
        base_point = gs.array([0.7, -0.2, 1.3])
        reference = PCA(n_components, whiten=True, svd_solver="full").fit(SAMPLES)

        if entry == "fit":
            model.fit(data, base_point=base_point)
            scores = model.transform(data)
        else:
            scores = model.fit_transform(data, base_point=base_point)

        np.testing.assert_allclose(
            scores.T @ scores / (len(data) - 1),
            np.eye(n_components),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            scores,
            _aligned_reference(model, reference, reference.transform(SAMPLES)),
            atol=1e-12,
        )
        np.testing.assert_allclose(model.transform(data), scores, atol=1e-12)
        np.testing.assert_allclose(
            model.transform(gs.array(QUERIES)),
            _aligned_reference(model, reference, reference.transform(QUERIES)),
            atol=1e-12,
        )

    @pytest.mark.parametrize("n_components", [1, 2, 3])
    def test_inverse_accepts_whitened_coordinates(self, n_components):
        reference = PCA(n_components, whiten=True, svd_solver="full").fit(SAMPLES)
        model = TangentPCA(Euclidean(3), n_components=n_components, whiten=True)
        model.fit(gs.array(SAMPLES), base_point=gs.array([0.7, -0.2, 1.3]))
        scores = _aligned_reference(model, reference, reference.transform(QUERIES))

        np.testing.assert_allclose(
            model.inverse_transform(gs.array(scores)),
            reference.inverse_transform(reference.transform(QUERIES)),
            atol=1e-12,
        )

    @pytest.mark.parametrize("n_components", [1, 2])
    def test_whitening_in_a_sphere_log_chart(self, n_components):
        space = Hypersphere(2)
        base_point = gs.array([0.0, 0.0, 1.0])
        tangent_vectors = gs.array(
            np.column_stack((SAMPLES[:, :2] / 10.0, np.zeros(len(SAMPLES))))
        )
        data = space.metric.exp(tangent_vectors, base_point)
        model = TangentPCA(space, n_components=n_components, whiten=True)
        plain = TangentPCA(space, n_components=n_components)
        scores = model.fit_transform(data, base_point=base_point)
        plain.fit(data, base_point=base_point)

        np.testing.assert_allclose(
            scores.T @ scores / (len(data) - 1),
            np.eye(n_components),
            atol=1e-12,
        )
        np.testing.assert_allclose(model.transform(data), scores, atol=1e-12)
        queries = space.metric.exp(
            gs.array([[0.12, -0.08, 0.0], [-0.21, 0.05, 0.0]]), base_point
        )
        np.testing.assert_allclose(
            model.inverse_transform(model.transform(queries)),
            plain.inverse_transform(plain.transform(queries)),
            atol=1e-12,
        )
        if n_components == 2:
            np.testing.assert_allclose(
                model.inverse_transform(scores), data, atol=1e-12
            )

    @pytest.mark.parametrize(
        "values",
        [
            np.full((5, 3), 2.0),
            np.array([[-3.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
            np.array(
                [
                    [1.0, 2.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 3.0, 0.0, 0.0],
                    [2.0, -1.0, 1.0, 0.0, 0.0],
                ]
            ),
        ],
        ids=["constant", "rank_one", "more_features_than_samples"],
    )
    def test_rank_deficient_coordinates_remain_finite(self, values):
        data = gs.array(values)
        space = Euclidean(values.shape[1])
        base_point = gs.zeros(values.shape[1])
        model = TangentPCA(space, whiten=True)
        plain = TangentPCA(space)
        scores = model.fit_transform(data, base_point=base_point)
        plain.fit(data, base_point=base_point)

        assert np.isfinite(scores).all()
        np.testing.assert_allclose(model.transform(data), scores, atol=1e-12)
        np.testing.assert_allclose(
            model.inverse_transform(scores),
            plain.inverse_transform(plain.transform(data)),
            atol=1e-12,
        )
        if np.all(values == values[0]):
            np.testing.assert_array_equal(scores, np.zeros_like(scores))

    def test_positive_variance_at_small_scale_is_preserved(self):
        data = gs.array(SAMPLES[:, :2] * 1e-14)
        model = TangentPCA(Euclidean(2), whiten=True)
        scores = model.fit_transform(data, base_point=gs.zeros(2))

        np.testing.assert_allclose(
            scores.T @ scores / (len(data) - 1), np.eye(2), atol=1e-12
        )
        np.testing.assert_allclose(model.inverse_transform(scores), data, atol=1e-26)

    def test_unwhitened_projection_is_unchanged(self):
        reference = PCA(n_components=2, svd_solver="full").fit(SAMPLES)
        model = TangentPCA(Euclidean(3), n_components=2)
        scores = model.fit_transform(gs.array(SAMPLES), base_point=gs.zeros(3))

        np.testing.assert_allclose(
            scores,
            _aligned_reference(model, reference, reference.transform(SAMPLES)),
            atol=1e-12,
        )

    @pytest.mark.parametrize("entry", ["fit", "fit_transform"])
    def test_whitening_requires_two_samples(self, entry):
        model = TangentPCA(Euclidean(3), whiten=True)
        with pytest.raises(ValueError, match="at least two samples"):
            getattr(model, entry)(gs.array([[1.0, 2.0, 3.0]]))
