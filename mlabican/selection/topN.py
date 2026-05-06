from typing import override

import numpy as np

from mlabican.selection.base import SelectionStrategy


class TopN(SelectionStrategy):
    """
    Top N instances selection strategy. This approach will rank all
    instances based on the highest confidence value and select the top N
    well ranked instances to label then in the current iteration.

    Args:
        threshold (int): The amount of instances that will be labeled
        in a single iteration. Default is 30.
    """

    @override
    def select_instances(
        self, probabilities: np.ndarray, threshold: int | float = 30, **kwargs
    ) -> np.ndarray:
        threshold = int(threshold)

        if threshold < 1:
            raise ValueError('Can not be zero or negative.')

        max_proba = np.max(probabilities, axis=1)

        return np.argsort(max_proba)[::-1][:threshold]
