from unittest import TestCase

import numpy as np

from mlabican.selection.rules import Rules
from mlabican.selection.base import SelectionStrategy
from mlabican.selection.threshold import Threshold
from mlabican.selection.topN import TopN


class StrategyNoImplementMock(SelectionStrategy):
    def select_instances(self, probabilities, threshold, **kwargs):
        super().select_instances(probabilities, threshold)


class StrategyImplementMock(SelectionStrategy):
    def select_instances(self, probabilities, threshold, **kwargs):
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
        self.kwargs = {}

    ###########################
    ##         BASE          ##
    ###########################
    def test_should_raise_type_error_when_try_to_call_select_instances_in_superclass(
        self,
    ):  # NOQA
        with self.assertRaises(TypeError) as err:
            SelectionStrategy()

    def test_should_raise_not_implemented_when_try_to_call_select_instances_in_superclass(
        self,
    ):  # NOQA
        with self.assertRaises(NotImplementedError) as err:
            StrategyNoImplementMock().select_instances(np.array([]), 10)

    def test_should_initiate_when_try_to_call_select_instances(self):
        self.assertTrue(
            StrategyImplementMock().select_instances(np.array([]), 10)
        )

    ###########################
    ##         TOP N         ##
    ###########################
    def test_topN_with_invalid_n_instances(self):
        with self.assertRaises(ValueError):
            strategy = TopN()
            strategy.select_instances(
                self.probabilities, 0, **self.kwargs
            )

    def test_top_n_selection(self):
        strategy = TopN()
        indices = strategy.select_instances(
            self.probabilities, 2, **self.kwargs
        )

        self.assertListEqual(indices.tolist(), [2, 0])
        self.assertListEqual(self.predictions[indices].tolist(), [0, 1])

        self.assertEqual(len(indices), 2)

    def test_top_n_selection_more_than_available_instances(self):
        strategy = TopN()
        indices = strategy.select_instances(
            self.probabilities, 20, **self.kwargs
        )

        self.assertListEqual(indices.tolist(), [2, 0, 4, 1, 3])
        self.assertListEqual(
            self.predictions[indices].tolist(),
            [0, 1, 1, 2, 0],
        )

        self.assertEqual(len(indices), 5)

    ###########################
    ##       Threshold       ##
    ###########################
    def test_simple_threshold_selection(self):
        strategy = Threshold()

        # Only instances with max_prob >= 0.75
        indices = strategy.select_instances(
            self.probabilities, .75, **self.kwargs
        )

        # Probs: 0.8 (idx 0), 0.9 (idx 2)
        self.assertListEqual(indices.tolist(), [0, 2])
        self.assertListEqual(self.predictions[indices].tolist(), [1, 0])

    ###########################
    ##         Rules         ##
    ###########################
    def test_rules_r1_with_threshold_selection(self):
        strategy = Rules()

        self.kwargs = {'prob_1_it': np.array(
            [
                [0.4, 0.3, 0.3],
                [0.3, 0.1, 0.6],
                [0.95, 0.03, 0.02],  # Selected by R1
                [0.8, 0.0, 0.2],
                [0.1, 0.9, 0.0],
            ]
        )}

        indices = strategy.select_instances(
            self.probabilities, .75, **self.kwargs
        )
        self.assertListEqual(indices.tolist(), [2])

    def test_rules_r2_with_threshold_selection(self):
        strategy = Rules()

        self.kwargs = {'prob_1_it': np.array(
            [
                [0.0, 0.8, 0.2],  # Selected by R2
                [0.4, 0.3, 0.3],  #
                [0.3, 0.1, 0.6],  #
                [0.4, 0.3, 0.3],  #
                [0.1, 0.9, 0.0],  #
            ]
        )}

        indices = strategy.select_instances(
            self.probabilities, .75, **self.kwargs
        )
        self.assertListEqual(indices.tolist(), [0])

    def test_rules_r3_with_threshold_selection(self):
        strategy = Rules()

        self.kwargs = {'prob_1_it': np.array(
            [
                [0.4, 0.3, 0.3],  #
                [0.3, 0.1, 0.6],  #
                [0.3, 0.1, 0.6],  #
                [0.2, 0.6, 0.2],  #
                [0.1, 0.0, 0.9],  # Selected by R3
            ]
        )}

        indices = strategy.select_instances(
            self.probabilities, .7, **self.kwargs
        )
        self.assertListEqual(indices.tolist(), [4])

    def test_rules_r4_with_threshold_selection(self):
        strategy = Rules()

        self.kwargs = {'prob_1_it': np.array(
            [
                [0.4, 0.3, 0.3],  # Selected by R4 by probs_x_it
                [0.3, 0.1, 0.6],  #
                [0.3, 0.3, 0.4],  # Selected by R4 by probs_x_it
                [0.0, 0.8, 0.2],  # Selected by R4
                [0.1, 0.0, 0.9],  # Selected by R4
            ]
        )}

        indices = strategy.select_instances(
            self.probabilities, .8, **self.kwargs
        )
        self.assertListEqual(indices.tolist(), [0, 2, 3, 4])

    def test_rules_no_instance_should_be_selected_threshold_selection(self):
        strategy = Rules()

        self.kwargs = {'prob_1_it': np.array(
            [
                [0.4, 0.3, 0.3],  # Selected by R4 by probs_x_it
                [0.3, 0.1, 0.6],  #
                [0.3, 0.3, 0.4],  # Selected by R4 by probs_x_it
                [0.0, 0.8, 0.2],  # Selected by R4
                [0.1, 0.0, 0.9],  # Selected by R4
            ]
        )}

        indices = strategy.select_instances(
            self.probabilities, 1.0, **self.kwargs
        )
        self.assertListEqual(indices.tolist(), [])
