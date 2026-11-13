import numpy as np

from mlabican.label.base import LabelingStrategy


class MemoryStrategy(LabelingStrategy):
    """
    Labels instances based on the historical classifier memory.
    """

    def label_instances(
        self, selected_indices: np.ndarray, **kwargs
    ) -> np.ndarray:
        cl_memory = kwargs.get('cl_memory', [])
        # Find the most frequent class in memory for each selected instance
        return np.array([np.argmax(cl_memory[x]) for x in selected_indices])
