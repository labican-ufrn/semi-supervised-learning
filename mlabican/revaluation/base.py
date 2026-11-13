from abc import ABC, abstractmethod

import numpy as np


class RevaluationStrategy(ABC):
    """Interface for revaluate instances strategies."""

    @abstractmethod
    def revaluate(
        self,
        instances: np.ndarray,
        labels: np.ndarray,
        labeled_mask: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Select the instances to may change the pseudo label from past
        iterations.

        Args:
            labeled_instances (np.ndarray): Current labeled instances.

        Raises:
            NotImplementedError: If you use superclass method.

        Returns:
            np.ndarray, np.ndarray: The modified label array and the modified
            labeled mask.
        """
        raise NotImplementedError('implement me!')
