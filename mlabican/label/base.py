from abc import ABC, abstractmethod

import numpy as np


class LabelingStrategy(ABC):
    """Interface for instance labeling strategies."""

    @abstractmethod
    def label_instances(
        self, selected_indices: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Assigns labels to the locally selected indices."""
        raise NotImplementedError('implement me!')
