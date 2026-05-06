import numpy as np

from mlabican.selection.base import SelectionStrategy


class Threshold(SelectionStrategy):
    """
    Selects any instance where the confidence is >= the current threshold.
    """

    def select_instances(
        self,
        probabilities: np.ndarray,
        threshold: int | float = 0.95,
        **kwargs,
    ) -> np.ndarray:
        max_proba = np.max(probabilities, axis=1)
        return np.where(max_proba >= threshold)[0]
