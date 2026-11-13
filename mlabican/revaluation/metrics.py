from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import silhouette_samples


class DifficultyMetric(ABC):
    """Interface for instance difficulty and cluster quality metrics."""

    @abstractmethod
    def calculate(
        self, instances: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Calculate difficulty metric scores for each labeled instance.

        Args:
            instances (np.ndarray): Feature array of labeled instances.
            labels (np.ndarray): Target labels of labeled instances.

        Raises:
            NotImplementedError: If called on the abstract base class.

        Returns:
            np.ndarray: Score for each labeled instance. Higher values
                indicate easier (better-placed) instances; lower values
                indicate harder (potentially mislabeled) ones.
        """
        raise NotImplementedError('implement me!')


class SilhouetteMetric(DifficultyMetric):
    """Calculates the Silhouette Coefficient for each labeled instance.

    Negative values typically indicate that an instance may have been
    assigned to an incorrect cluster or pseudo-label.
    """

    def calculate(
        self, instances: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        unique_labels = np.unique(labels)
        # Silhouette requires at least 2 distinct clusters and at least 2 samples
        if len(unique_labels) < 2 or len(labels) <= len(unique_labels):
            return np.zeros(len(labels), dtype=float)

        return silhouette_samples(instances, labels)


class DaviesBouldinMetric(DifficultyMetric):
    """Projects the Davies-Bouldin index to per-instance difficulty scores.

    The Davies-Bouldin index is a global cluster quality measure. To
    produce a per-instance score compatible with the ``DifficultyMetric``
    interface we compute each instance's normalised distance to its own
    cluster centroid divided by the cluster's mean intra-cluster distance.
    A higher returned value means the instance is closer to its centroid
    (easier); a lower value means the instance is far from its centroid
    relative to the cluster spread (harder / potentially mislabeled).

    Note:
        Because the score is inverted (higher → easier) the same
        ``threshold`` convention applies: values below the threshold are
        considered weak pseudo-labels.
    """

    def calculate(
        self, instances: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        unique_labels = np.unique(labels)
        # Requires at least 2 clusters to compute meaningful inter-cluster distances
        if len(unique_labels) < 2 or len(labels) <= len(unique_labels):
            return np.zeros(len(labels), dtype=float)

        scores = np.zeros(len(labels), dtype=float)
        centroids = {}
        mean_dists = {}

        # Compute centroid and mean intra-cluster distance for each cluster
        for label in unique_labels:
            mask = labels == label
            cluster_pts = instances[mask]
            centroid = cluster_pts.mean(axis=0)
            centroids[label] = centroid
            dists = np.linalg.norm(cluster_pts - centroid, axis=1)
            mean_dists[label] = dists.mean() if len(dists) > 1 else 1.0

        # Per-instance score: normalised proximity to own centroid
        # score = 1 - (d_to_centroid / mean_intra_dist), clipped to [-1, 1]
        for label in unique_labels:
            mask = labels == label
            cluster_pts = instances[mask]
            centroid = centroids[label]
            dists = np.linalg.norm(cluster_pts - centroid, axis=1)
            normalised = dists / (mean_dists[label] + 1e-10)
            # Convert: small distance (good) → positive score; large → negative
            scores[mask] = 1.0 - normalised

        return np.clip(scores, -1.0, 1.0)
