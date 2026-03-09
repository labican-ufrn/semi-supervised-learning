from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from mlabican.ensemble.ensemble import Ensemble


class TestEnsemble(TestCase):
    def setUp(self):
        # Create mock classifiers that satisfy validation
        self.mock_c1 = MagicMock()
        self.mock_c1.predict.return_value = np.array([1])
        self.mock_c1.predict_proba.return_value = np.array([0.8])

        self.mock_c2 = MagicMock()
        self.mock_c2.predict.return_value = np.array([0])
        self.mock_c2.predict_proba.return_value = np.array([0.6])

        self.classifiers = [self.mock_c1, self.mock_c2]
        self.X = np.array([[5.1, 3.5]])
        self.y = np.array([1])

    def test_create_ensemble_without_classifiers(self) -> None:
        ensemble = Ensemble()
        self.assertListEqual(ensemble.ensemble, [])

    def test_should_raise_exception_when_classifier_missing_methods(self) -> None:
        # A mock that lacks predict_proba
        invalid_classifier = MagicMock()
        del invalid_classifier.predict_proba

        with self.assertRaises(AttributeError):
            Ensemble([invalid_classifier])

        invalid_classifier_2 = MagicMock()
        del invalid_classifier_2.predict

        with self.assertRaises(AttributeError):
            Ensemble([invalid_classifier_2])

        invalid_classifier_3 = MagicMock()
        del invalid_classifier_3.predict
        del invalid_classifier_3.predict_proba

        with self.assertRaises(AttributeError):
            Ensemble([invalid_classifier_3])

    def test_fit_ensemble_calls_fit_on_all_members(self) -> None:
        ensemble = Ensemble(self.classifiers)
        ensemble.fit_ensemble(self.X, self.y)

        self.mock_c1.fit.assert_called_once_with(self.X, self.y)
        self.mock_c2.fit.assert_called_once_with(self.X, self.y)

    def test_predict_majority_voting(self) -> None:
        # Mock 3 classifiers: two predict 1, one predicts 0. Mode should be 1.
        c3 = MagicMock()
        c3.predict.return_value = np.array([1])

        ensemble = Ensemble([self.mock_c1, self.mock_c2, c3])
        predictions = ensemble.predict(self.X)

        self.assertEqual(predictions[0], 1)
        self.assertIsInstance(predictions, np.ndarray)

    def test_drop_ensemble(self) -> None:
        ensemble = Ensemble(self.classifiers)
        ensemble.drop_ensemble()
        self.assertEqual(len(ensemble.ensemble), 0)
