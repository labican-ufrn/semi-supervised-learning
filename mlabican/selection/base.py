from abc import ABC, abstractmethod

import numpy as np

from mlabican.utils import get_logger


class SelectionStrategy(ABC):
    """Interface for instance selection strategies."""

    def __init__(self, verbose: bool = False):
        self.logger = get_logger()
        self.verbose = verbose

    @abstractmethod
    def select_instances(
        self, probabilities: np.ndarray, threshold: float, **kwargs
    ) -> np.ndarray:
        """Method to select the instances based on some criteria,
        such as:
            - Amount of instances;
            - Threshold;
            - Threshold with rules;

        Args:
            probabilities (np.ndarray): probabilities of each label in
                the current iteration.
            threshold (float): Number of instances or threshold to
                select the unlabeled instances.
            kwargs (dict): A dictionary with parameters that can be used
                in classes that inherit from this one. Options include:
                # Rules
                - pred_1_it (dict): Prediction that was made in the first iteration.


        Raises:
            NotImplementedError: If you use superclass method.

        Returns:
            np.ndarray: An array of selected local indices.
        """
        raise NotImplementedError('implement me!')
