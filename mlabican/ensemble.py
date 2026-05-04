from statistics import mode

import numpy as np

from mlabican.utils import has_method


class Ensemble:
    """Initializes the Ensemble with a list of base classifiers.

    Args:
        classifiers (list | None): A list of classifier objects. Each
            classifier must implement 'predict' and 'predict_proba'
            methods. Defaults to None.

    Raises:
        AttributeError: If any provided classifier lacks the 'predict'
            or 'predict_proba' methods.
    """

    def __init__(
        self,
        classifiers: list | None = None,
    ) -> None:
        if classifiers is not None:
            if not self._validate_classifiers(classifiers):
                raise AttributeError(
                    'Any classifier does not have "predict" or "predict_proba" methods.'
                )
        self.ensemble = classifiers or []

    def _validate_classifiers(self, classifiers: list) -> bool:
        """
        Validate if the base classifiers have the predict and
        predict_proba methods.

        Args:
            classifiers (list): A list of classifier objects to validate.

        Returns:
            bool: True if all classifiers have 'predict' and
                'predict_proba' methods, False otherwise.
        """
        return all(
            (has_method(c, 'predict') and has_method(c, 'predict_proba'))
            for c in classifiers
        )

    def drop(self) -> None:
        """Clears all classifiers currently stored in the ensemble."""
        self.ensemble = []

    def fit(self, instances: np.ndarray, labels: np.ndarray) -> None:
        """
        Trains all classifiers currently stored in the ensemble.

        Args:
            instances (np.ndarray): The training data features.
            labels (np.ndarray): The target labels for training.
        """
        for classifier in self.ensemble:
            self.fit_single_classifier(classifier, instances, labels)

    def fit_single_classifier(
        self,
        classifier,
        instances: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Trains a single classifier on the provided data.

        Args:
            classifier: The classifier instance to train.
            instances (np.ndarray): The training data features.
            labels (np.ndarray): The target labels for training.
        """
        classifier.fit(instances, labels)

    def predict(self, instances: np.ndarray) -> np.ndarray:
        """
        Predicts the most common label (mode) among all classifiers for
        each instance.

        Args:
            instances (np.ndarray): The data features to predict.

        Returns:
            np.ndarray: An array of the majority-voted predicted labels
            for each instance.
        """
        y_pred = np.array([], dtype='int64')

        for instance in instances:
            pred = []

            for classifier in self.ensemble:
                pred.append(
                    classifier.predict(instance.reshape(1, -1)).tolist()[0]
                )
            y_pred = np.append(y_pred, mode(pred))

        return y_pred

    def predict_proba(self, instances: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities of the instances for all classifiers.

        Args:
            instances (np.ndarray): The data features to predict.

        Returns:
            np.ndarray: An array of the majority-voted predicted labels
            for each instance.
        """
        y_pred = np.array([], dtype='float64')

        for instance in instances:
            probas = []

            for classifier in self.ensemble:
                probas.append(
                    classifier.predict_proba(instance.reshape(1, -1)).tolist()
                )
            y_probas = np.average(probas, axis=0)
            y_pred = np.append(y_pred, y_probas)

        return y_pred
