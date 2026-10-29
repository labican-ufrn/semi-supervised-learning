import numpy as np

from mlabican.revaluation.base import RevaluationStrategy
from mlabican.revaluation.metrics import DifficultyMetric, SilhouetteMetric


class DeferredRevaluationStrategy(RevaluationStrategy):
    """Revaluation strategy that returns weak pseudo-labels to the unlabeled pool.

    Evaluates every pseudo-labeled instance using a ``DifficultyMetric``.
    Instances whose score falls below ``threshold`` are considered noisy or
    uncertain: their label is reset to ``-1`` and they are removed from the
    labeled mask so that the FlexCon loop can re-label them in a later
    iteration.

    This is a *deferred* strategy because the final label correction happens
    in a future iteration (the instance re-enters the unlabeled pool).

    Args:
        metric (DifficultyMetric, optional): Metric used to evaluate instance
            quality. Defaults to ``SilhouetteMetric()``.
        threshold (float, optional): Score below which an instance is
            considered a weak pseudo-label. Defaults to ``-0.2``.
        protect_initial (bool, optional): If ``True``, instances labeled in
            iteration 0 (ground-truth labels) are never removed.
            Defaults to ``True``.
    """

    def __init__(
        self,
        metric: DifficultyMetric | None = None,
        threshold: float = -0.2,
        protect_initial: bool = True,
    ) -> None:
        self.metric = metric or SilhouetteMetric()
        self.threshold = threshold
        self.protect_initial = protect_initial

    def revaluate(
        self,
        instances: np.ndarray,
        labels: np.ndarray,
        labeled_mask: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset weak pseudo-labels to ``-1`` and move them to the unlabeled pool.

        Args:
            instances (np.ndarray): Full dataset feature array.
            labels (np.ndarray): Current transduction label array
                (``-1`` for unlabeled instances).
            labeled_mask (np.ndarray): Boolean mask of currently labeled
                instances.
            **kwargs:
                labeled_iter (np.ndarray): Iteration in which each instance
                    was labeled (``0`` for ground-truth labels). Required when
                    ``protect_initial=True``.

        Returns:
            tuple[np.ndarray, np.ndarray]: Updated labels and labeled mask.
        """
        labeled_indices = np.where(labeled_mask)[0]
        if labeled_indices.size == 0:
            return labels, labeled_mask

        X_labeled = instances[labeled_indices]
        y_labeled = labels[labeled_indices]

        if len(np.unique(y_labeled)) < 2:
            return labels, labeled_mask

        scores = self.metric.calculate(X_labeled, y_labeled)
        weak_mask = scores < self.threshold

        if self.protect_initial:
            labeled_iter = kwargs.get('labeled_iter')
            if labeled_iter is not None:
                initial_labeled = labeled_iter[labeled_indices] == 0
                weak_mask[initial_labeled] = False

        if not np.any(weak_mask):
            return labels, labeled_mask

        indices_weak = labeled_indices[weak_mask]
        labels[indices_weak] = -1
        labeled_mask[indices_weak] = False

        return labels, labeled_mask


class ImmediateRevaluationStrategy(RevaluationStrategy):
    """Revaluation strategy that corrects weak pseudo-labels in the same step.

    Evaluates every pseudo-labeled instance using a ``DifficultyMetric``.
    Instances whose score falls below ``threshold`` are considered uncertain
    and are immediately re-labeled via weighted voting between the base
    estimator and a committee model. Re-labeled instances are then
    *protected* from future removal to avoid oscillation.

    This is an *immediate* strategy because the label is corrected in the
    current iteration rather than being deferred to a future one.

    Args:
        metric (DifficultyMetric, optional): Metric used to evaluate instance
            quality. Defaults to ``SilhouetteMetric()``.
        threshold (float, optional): Score below which an instance is
            considered a weak pseudo-label. Defaults to ``-0.2``.
        committee_weight (float, optional): Weight given to the committee
            model's probability estimates during weighted voting.
            Defaults to ``0.51``.
        protect_initial (bool, optional): If ``True``, instances labeled in
            iteration 0 (ground-truth labels) are never re-labeled.
            Defaults to ``True``.
    """

    def __init__(
        self,
        metric: DifficultyMetric | None = None,
        threshold: float = -0.2,
        committee_weight: float = 0.51,
        protect_initial: bool = True,
    ) -> None:
        self.metric = metric or SilhouetteMetric()
        self.threshold = threshold
        self.committee_weight = committee_weight
        self.protect_initial = protect_initial
        self.protected_indices_: set[int] = set()

    def reset(self) -> None:
        """Reset protected indices between training runs."""
        self.protected_indices_.clear()

    def revaluate(
        self,
        instances: np.ndarray,
        labels: np.ndarray,
        labeled_mask: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-label weak pseudo-labeled instances using a committee model.

        Args:
            instances (np.ndarray): Full dataset feature array.
            labels (np.ndarray): Current transduction label array.
            labeled_mask (np.ndarray): Boolean mask of currently labeled
                instances.
            **kwargs:
                estimator: Base classifier with ``predict_proba`` and
                    ``classes_`` attributes. **Required**.
                committee: Ensemble/committee model with ``predict_proba``.
                    **Required**.
                labeled_iter (np.ndarray): Iteration in which each instance
                    was labeled. Required when ``protect_initial=True``.

        Returns:
            tuple[np.ndarray, np.ndarray]: Updated labels and labeled mask.

        Raises:
            ValueError: If ``estimator`` or ``committee`` are not provided.
        """
        estimator = kwargs.get('estimator')
        committee = kwargs.get('committee')
        if estimator is None or committee is None:
            raise ValueError(
                'ImmediateRevaluationStrategy requires both `estimator` and '
                '`committee` keyword arguments.'
            )

        labeled_indices = np.where(labeled_mask)[0]
        if labeled_indices.size == 0:
            return labels, labeled_mask

        X_labeled = instances[labeled_indices]
        y_labeled = labels[labeled_indices]

        if len(np.unique(y_labeled)) < 2:
            return labels, labeled_mask

        scores = self.metric.calculate(X_labeled, y_labeled)
        weak_mask = scores < self.threshold

        # Exclude already-protected instances (corrected in past iterations)
        if self.protected_indices_:
            is_protected = np.isin(labeled_indices, list(self.protected_indices_))
            weak_mask[is_protected] = False

        if self.protect_initial:
            labeled_iter = kwargs.get('labeled_iter')
            if labeled_iter is not None:
                initial_labeled = labeled_iter[labeled_indices] == 0
                weak_mask[initial_labeled] = False

        if not np.any(weak_mask):
            return labels, labeled_mask

        indices_weak = labeled_indices[weak_mask]
        X_weak = instances[indices_weak]

        prob_estimator = estimator.predict_proba(X_weak)
        prob_committee = committee.predict_proba(X_weak)

        classes = getattr(estimator, 'classes_', np.unique(y_labeled))
        final_prob = (
            (1.0 - self.committee_weight) * prob_estimator
            + self.committee_weight * prob_committee
        )
        new_labels = classes[np.argmax(final_prob, axis=1)]

        labels[indices_weak] = new_labels
        labeled_mask[indices_weak] = True
        self.protected_indices_.update(indices_weak.tolist())

        return labels, labeled_mask
