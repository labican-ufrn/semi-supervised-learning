from unittest import TestCase

import numpy as np
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier as Naive

from mlabican.flexcon import FlexCon
from mlabican.revaluation.base import RevaluationStrategy
from mlabican.revaluation.metrics import (
    DaviesBouldinMetric,
    DifficultyMetric,
    SilhouetteMetric,
)
from mlabican.revaluation.strategies import (
    DeferredRevaluationStrategy,
    ImmediateRevaluationStrategy,
)
from mlabican.selection.threshold import Threshold


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class StrategyNoImplementMock(RevaluationStrategy):
    def revaluate(
        self,
        instances: np.ndarray,
        labels: np.ndarray,
        labeled_mask: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        return super().revaluate(instances, labels, labeled_mask)


class StrategyImplementMock(RevaluationStrategy):
    def revaluate(
        self,
        instances: np.ndarray,
        labels: np.ndarray,
        labeled_mask: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.array([True]), np.array([True])


class MetricNoImplementMock(DifficultyMetric):
    def calculate(
        self, instances: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        return super().calculate(instances, labels)


class DummyEstimator:
    classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Always votes for class 1
        return np.array([[0.1, 0.9] for _ in range(len(X))])


class DummyCommittee:
    def __init__(self, probas: np.ndarray):
        self.probas = probas

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.probas[: len(X)]


# ---------------------------------------------------------------------------
# Shared dataset helper
# ---------------------------------------------------------------------------


def _two_cluster_dataset():
    """Three well-separated points in cluster 0, three in cluster 1,
    and one point near cluster 0 that is mislabeled as class 1 (index 6)."""
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.0, 0.2],
        [10.0, 10.0],
        [10.1, 10.1],
        [10.0, 10.2],
        [0.05, 0.05],  # Near cluster 0, mislabeled as 1
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    mask = np.ones(len(y), dtype=bool)
    return X, y, mask


class RevaluationTestCase(TestCase):
    # ------------------------------------------------------------------
    # RevaluationStrategy ABC
    # ------------------------------------------------------------------

    def test_should_raise_type_error_when_try_create_revaluation_without_implementing_methods(
        self,
    ):
        with self.assertRaises(TypeError):
            RevaluationStrategy()

    def test_should_raise_not_implemented_when_calling_base_revaluate(self):
        with self.assertRaises(NotImplementedError):
            StrategyNoImplementMock().revaluate(
                np.array([]), np.array([]), np.array([])
            )

    def test_should_execute_revaluate_on_concrete_mock(self):
        labels, mask = StrategyImplementMock().revaluate(
            np.array([]), np.array([]), np.array([])
        )
        self.assertTrue(labels[0])
        self.assertTrue(mask[0])

    # ------------------------------------------------------------------
    # DifficultyMetric ABC
    # ------------------------------------------------------------------

    def test_should_raise_type_error_on_abstract_difficulty_metric(self):
        with self.assertRaises(TypeError):
            DifficultyMetric()

    def test_should_raise_not_implemented_when_calling_base_calculate(self):
        with self.assertRaises(NotImplementedError):
            MetricNoImplementMock().calculate(np.array([]), np.array([]))

    # ------------------------------------------------------------------
    # SilhouetteMetric
    # ------------------------------------------------------------------

    def test_silhouette_metric_with_insufficient_classes_returns_zeros(self):
        metric = SilhouetteMetric()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 0])
        scores = metric.calculate(X, y)
        self.assertEqual(len(scores), 2)
        self.assertTrue(np.all(scores == 0.0))

    def test_silhouette_metric_with_two_separated_clusters(self):
        metric = SilhouetteMetric()
        X = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ])
        y = np.array([0, 0, 1, 1])
        scores = metric.calculate(X, y)
        self.assertEqual(len(scores), 4)
        self.assertTrue(np.all(scores > 0.5))

    # ------------------------------------------------------------------
    # DaviesBouldinMetric
    # ------------------------------------------------------------------

    def test_davies_bouldin_metric_with_insufficient_classes_returns_zeros(self):
        metric = DaviesBouldinMetric()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 0])
        scores = metric.calculate(X, y)
        self.assertEqual(len(scores), 2)
        self.assertTrue(np.all(scores == 0.0))

    def test_davies_bouldin_metric_centroids_score_highest(self):
        """Point closest to its cluster centroid scores higher than a far outlier."""
        metric = DaviesBouldinMetric()
        # 3 points per cluster: indices 0-1 near centroid, index 2 is outlier
        X = np.array([
            [0.0, 0.0],   # cluster 0 – near centroid
            [0.1, 0.0],   # cluster 0 – near centroid
            [5.0, 0.0],   # cluster 0 – clear outlier
            [20.0, 0.0],  # cluster 1 – near centroid
            [20.1, 0.0],  # cluster 1 – near centroid
            [25.0, 0.0],  # cluster 1 – clear outlier
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        scores = metric.calculate(X, y)
        self.assertEqual(len(scores), 6)
        # Near-centroid points should score higher than the outlier
        self.assertGreater(scores[0], scores[2])
        self.assertGreater(scores[3], scores[5])

    def test_davies_bouldin_metric_returns_scores_in_valid_range(self):
        metric = DaviesBouldinMetric()
        X, y, _ = _two_cluster_dataset()
        scores = metric.calculate(X, y)
        self.assertTrue(np.all(scores >= -1.0))
        self.assertTrue(np.all(scores <= 1.0))

    # ------------------------------------------------------------------
    # DeferredRevaluationStrategy
    # ------------------------------------------------------------------

    def test_deferred_returns_unchanged_when_no_labeled(self):
        strategy = DeferredRevaluationStrategy()
        X = np.zeros((4, 2))
        y = np.array([-1, -1, -1, -1])
        mask = np.zeros(4, dtype=bool)
        new_y, new_mask = strategy.revaluate(X, y, mask)
        np.testing.assert_array_equal(new_y, y)
        np.testing.assert_array_equal(new_mask, mask)

    def test_deferred_removes_weak_instance_to_unlabeled_pool(self):
        strategy = DeferredRevaluationStrategy(threshold=-0.1, protect_initial=False)
        X, y, mask = _two_cluster_dataset()
        new_y, new_mask = strategy.revaluate(X, y.copy(), mask.copy())

        # Mislabeled index 6 should be returned to the unlabeled pool
        self.assertFalse(new_mask[6])
        self.assertEqual(new_y[6], -1)
        # Well-placed instances must remain labeled
        self.assertTrue(new_mask[0])
        self.assertTrue(new_mask[3])

    def test_deferred_protects_initial_ground_truth_instances(self):
        strategy = DeferredRevaluationStrategy(threshold=-0.1, protect_initial=True)
        X, y, mask = _two_cluster_dataset()
        labeled_iter = np.zeros(len(y), dtype=int)  # All marked as initial

        new_y, new_mask = strategy.revaluate(
            X, y.copy(), mask.copy(), labeled_iter=labeled_iter
        )

        # Index 6 is mislabeled but protected as initial → must NOT be removed
        self.assertTrue(new_mask[6])
        self.assertEqual(new_y[6], 1)

    def test_deferred_accepts_davies_bouldin_metric(self):
        strategy = DeferredRevaluationStrategy(
            metric=DaviesBouldinMetric(),
            threshold=-0.1,
            protect_initial=False,
        )
        X, y, mask = _two_cluster_dataset()
        new_y, new_mask = strategy.revaluate(X, y.copy(), mask.copy())
        # Should run without error and return arrays of correct shape
        self.assertEqual(len(new_y), len(y))
        self.assertEqual(len(new_mask), len(mask))

    # ------------------------------------------------------------------
    # ImmediateRevaluationStrategy
    # ------------------------------------------------------------------

    def test_immediate_raises_when_estimator_not_provided(self):
        strategy = ImmediateRevaluationStrategy(threshold=-0.1)
        X, y, mask = _two_cluster_dataset()
        committee = DummyCommittee(np.array([[0.95, 0.05]] * 10))
        with self.assertRaises(ValueError):
            strategy.revaluate(X, y.copy(), mask.copy(), committee=committee)

    def test_immediate_raises_when_committee_not_provided(self):
        strategy = ImmediateRevaluationStrategy(threshold=-0.1)
        X, y, mask = _two_cluster_dataset()
        with self.assertRaises(ValueError):
            strategy.revaluate(
                X, y.copy(), mask.copy(), estimator=DummyEstimator()
            )

    def test_immediate_returns_unchanged_when_no_labeled(self):
        strategy = ImmediateRevaluationStrategy()
        X = np.zeros((4, 2))
        y = np.array([-1, -1, -1, -1])
        mask = np.zeros(4, dtype=bool)
        committee = DummyCommittee(np.array([[0.95, 0.05]] * 10))
        new_y, new_mask = strategy.revaluate(
            X, y, mask, estimator=DummyEstimator(), committee=committee
        )
        np.testing.assert_array_equal(new_y, y)
        np.testing.assert_array_equal(new_mask, mask)

    def test_immediate_relabels_weak_instance_via_committee(self):
        # Committee strongly votes for class 0; specialist votes for class 1
        committee = DummyCommittee(np.array([[0.95, 0.05]] * 10))
        strategy = ImmediateRevaluationStrategy(
            threshold=-0.1,
            committee_weight=0.6,
            protect_initial=False,
        )
        X, y, mask = _two_cluster_dataset()
        new_y, new_mask = strategy.revaluate(
            X, y.copy(), mask.copy(),
            estimator=DummyEstimator(),
            committee=committee,
        )

        # Mislabeled index 6 should be corrected to class 0 by the committee
        self.assertTrue(new_mask[6])
        self.assertEqual(new_y[6], 0)
        self.assertIn(6, strategy.protected_indices_)

    def test_immediate_protects_initial_ground_truth_instances(self):
        committee = DummyCommittee(np.array([[0.95, 0.05]] * 10))
        strategy = ImmediateRevaluationStrategy(
            threshold=-0.1, protect_initial=True
        )
        X, y, mask = _two_cluster_dataset()
        labeled_iter = np.zeros(len(y), dtype=int)

        new_y, new_mask = strategy.revaluate(
            X, y.copy(), mask.copy(),
            estimator=DummyEstimator(),
            committee=committee,
            labeled_iter=labeled_iter,
        )

        # Index 6 is mislabeled but protected as initial → label unchanged
        self.assertEqual(new_y[6], 1)

    def test_immediate_reset_clears_protected_indices(self):
        strategy = ImmediateRevaluationStrategy(
            threshold=-0.1, protect_initial=False
        )
        committee = DummyCommittee(np.array([[0.95, 0.05]] * 10))
        X, y, mask = _two_cluster_dataset()

        strategy.revaluate(
            X, y.copy(), mask.copy(),
            estimator=DummyEstimator(),
            committee=committee,
        )
        self.assertGreater(len(strategy.protected_indices_), 0)

        strategy.reset()
        self.assertEqual(len(strategy.protected_indices_), 0)

    # ------------------------------------------------------------------
    # Integration: FlexCon + DeferredRevaluationStrategy
    # ------------------------------------------------------------------

    def test_flexcon_integration_with_deferred_revaluation(self):
        iris = load_iris()
        X = iris.data
        y = iris.target.copy()
        rng = np.random.RandomState(42)
        y[rng.rand(len(y)) < 0.4] = -1

        model = FlexCon(
            estimator=Naive(),
            selection_strategy=Threshold(),
            revaluation_strategy=DeferredRevaluationStrategy(threshold=-0.1),
            threshold=0.95,
            max_iter=10,
            verbose=False,
        )
        model.fit(X, y)
        self.assertGreater(model.n_iter_, 0)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))
