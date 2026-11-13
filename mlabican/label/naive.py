import numpy as np

from mlabican.label.base import LabelingStrategy


class NaiveStrategy(LabelingStrategy):
    """
    Directly labels instances based on the classifier's standard prediction.
    """

    def label_instances(
        self, selected_indices: np.ndarray, **kwargs
    ) -> np.ndarray:
        predictions = kwargs.get('predictions', [])
        return predictions[selected_indices]
