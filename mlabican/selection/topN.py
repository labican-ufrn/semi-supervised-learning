from typing import override

import numpy as np

from mlabican.selection.selection import SelectionStrategy


class TopN(SelectionStrategy):
    """
    Top N instances selection strategy. This approach will rank all
    instances based on the highest confidence value and select the top N
    well ranked instances to label then in the current iteration.

    Args:
        n_instances (int): The amount of instances that will be labeled
        in a single iteration. Default is 30.
    """

    def __init__(self, n_instances: int = 30) -> None:
        super().__init__()
        if n_instances < 1:
            raise ValueError('Can not be negative.')

        self.n_instances = n_instances

    @override
    def select_instances(
        self, probabilities: np.ndarray, **kwargs
    ) -> np.ndarray:
        max_proba = np.max(probabilities, axis=1)

        # Sort indices by descending probability and take top N
        top_n_indices = np.argsort(max_proba)[::-1][: self.n_instances]

        return top_n_indices
