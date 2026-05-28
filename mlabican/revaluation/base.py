from abc import ABC, abstractmethod

import numpy as np


class RevaluationStrategy(ABC):
    """Interface for revaluate instances strategies."""

    @abstractmethod
    def revaluate(self, labeled_instances: np.ndarray, **kwargs) -> np.ndarray:
        """
        Select the instances to may change the pseudo label from past
        iterations.

        Args:
            labeled_instances (np.ndarray): Current labeled instances.

        Raises:
            NotImplementedError: If you use superclass method.

        Returns:
            np.ndarray: The instances where the label should be changed.
        """
        raise NotImplementedError('implement me!')
