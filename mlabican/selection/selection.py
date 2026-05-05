from abc import ABC, abstractmethod

import numpy as np


class SelectionStrategy(ABC):
    """Interface for instance selection strategies."""

    @abstractmethod
    def select_instances(
        self, probabilities: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Method to select the instances based on some criteria,
        such as:
            - Amount of instances;
            - Threshold;
            - Threshold with rules;

        Args:
            probabilities (np.ndarray): probabilities of each label in
                the current iteration.
            kwargs (dict): A dictionary with parameters that can be used
                in classes that inherit from this one. Options include:
                # Thresholds
                - threshold (float): The threshold for the current iteration.
                # Rules
                - pred_1_it (dict): Prediction that was made in the first iteration.
                - cl_memory (dict): Classification memory.


        Raises:
            NotImplementedError: If you use superclass method.

        Returns:
            np.ndarray: An array of selected local indices.
        """
        raise NotImplementedError('implement me!')
