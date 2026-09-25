from unittest import TestCase

import numpy as np

from mlabican.revaluation.base import RevaluationStrategy


class StrategyNoImplementMock(RevaluationStrategy):
    def revaluate(
        self,
        instances: np.ndarray,
        labels: np.ndarray,
        labeled_mask: np.ndarray,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        return super().revaluate(instances, labels, labeled_mask)


class StrategyImplementMock(RevaluationStrategy):
    def revaluate(
        self,
        instances: np.ndarray,
        labels: np.ndarray,
        labeled_mask: np.ndarray,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.array([True]), np.array([True])



class RevaluationTestCase(TestCase):
    ###########################
    ##         BASE          ##
    ###########################
    def test_should_raise_type_error_when_try_create_threshold_object_without_implement_abstract_methods(
        self,
    ):  # NOQA
        with self.assertRaises(TypeError) as err:
            RevaluationStrategy()

    def test_should_raise_not_implemented_when_try_to_call_revaluate_in_superclass(
        self,
    ):  # NOQA
        with self.assertRaises(NotImplementedError) as err:
            StrategyNoImplementMock().revaluate([], [], [])

    def test_should_initiate_when_try_to_call_select_instances(self):
        self.assertTrue(StrategyImplementMock().revaluate([], [], []))
