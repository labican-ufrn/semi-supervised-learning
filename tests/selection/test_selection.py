from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from mlabican.selection.selection import SelectionStrategy
from mlabican.selection.topN import TopN


class StrategyNoImplementMock(SelectionStrategy, MagicMock):
    def select_instances(self, probabilities, **kwargs):
        super().select_instances(probabilities)


class StrategyImplementMock(SelectionStrategy, MagicMock):
    def select_instances(self, probabilities, **kwargs):
        return True


class TestSelectionStrategy(TestCase):
    def setUp(self):
        # 5 instances, 3 classes
        self.probabilities = np.array(
            [
                [0.1, 0.8, 0.1],  # Max 0.8 (Class 1)
                [0.2, 0.2, 0.6],  # Max 0.6 (Class 2)
                [0.9, 0.05, 0.05],  # Max 0.9 (Class 0)
                [0.4, 0.4, 0.2],  # Max 0.4 (Class 0 or 1)
                [0.3, 0.7, 0.0],  # Max 0.7 (Class 1)
            ]
        )
        self.predictions = np.array([1, 2, 0, 0, 1])
        self.kwargs = {'threshold': 0.95}

    def test_should_raise_type_error_when_try_to_call_select_instances_in_superclass(
        self,
    ):  # NOQA
        with self.assertRaises(TypeError) as err:
            SelectionStrategy()

    def test_should_raise_not_implemented_when_try_to_call_select_instances_in_superclass(
        self,
    ):  # NOQA
        with self.assertRaises(NotImplementedError) as err:
            StrategyNoImplementMock().select_instances(np.array([]))

    def test_should_initiate_when_try_to_call_select_instances(self):
        self.assertTrue(StrategyImplementMock().select_instances(np.array([])))

    ###########################
    ##      TOP N Tests      ##
    ###########################
    def test_topN_with_invalid_n_instances(self):
        with self.assertRaises(ValueError):
            TopN(n_instances=0)

    def test_top_n_selection(self):
        strategy = TopN(n_instances=2)
        indices = strategy.select_instances(
            self.probabilities, **self.kwargs
        )

        self.assertListEqual(indices.tolist(), [2, 0])
        self.assertListEqual(self.predictions[indices].tolist(), [0, 1])

        self.assertEqual(len(indices), 2)

    def test_top_n_selection_more_than_available_instances(self):
        strategy = TopN(n_instances=20)
        indices = strategy.select_instances(
            self.probabilities, **self.kwargs
        )

        self.assertListEqual(indices.tolist(), [2, 0, 4, 1, 3])
        self.assertListEqual(
            self.predictions[indices].tolist(),
            [0, 1, 1, 2, 0],
        )

        self.assertEqual(len(indices), 5)
