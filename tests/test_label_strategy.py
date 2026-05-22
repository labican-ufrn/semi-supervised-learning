from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from mlabican.label.base import LabelingStrategy
from mlabican.label.memory import MemoryStrategy
from mlabican.label.naive import NaiveStrategy
from mlabican.label.rules import RuleBasedLabelStrategy


class StrategyNoImplementMock(LabelingStrategy):
    def label_instances(self, selected_indices: np.ndarray, **kwargs):
        super().label_instances(selected_indices, **kwargs)


class StrategyImplementMock(LabelingStrategy):
    def label_instances(self, selected_indices: np.ndarray, **kwargs):
        return True


class TestLabelingStrategy(TestCase):
    def setUp(self):
        # Common data setup for the tests

        # 1. Base Predictions (used by Naive Strategy)
        # Assume 4 total instances in the current batch
        self.predictions = np.array([2, 0, 1, 0])

        # 2. Classifier Memory (used by Memory and Rule-Based Strategies)
        # Represents the weight/count of times an instance was classified as a specific class
        self.cl_memory = [
            [1.0, 0.0, 5.0],  # Instance 0: Most likely Class 2
            [3.0, 1.0, 0.0],  # Instance 1: Most likely Class 0
            [0.0, 4.0, 0.0],  # Instance 2: Most likely Class 1
            [2.0, 2.0, 6.0],  # Instance 3: Most likely Class 2
        ]

        # 3. Iteration Predictions (used by Rule-Based Strategy)
        self.pred_1_it = {
            0: {"classes": 2, "confidence": 0.95},
            1: {"classes": 0, "confidence": 0.80},
            2: {"classes": 1, "confidence": 0.90},
            3: {"classes": 1, "confidence": 0.85}, # Notice class 1 here
        }

        self.pred_x_it = {
            0: {"classes": 2, "confidence": 0.98}, # MATCHES pred_1_it (Class 2)
            1: {"classes": 0, "confidence": 0.70}, # MATCHES pred_1_it (Class 0)
            2: {"classes": 2, "confidence": 0.88}, # DIFFERS from pred_1_it (2 vs 1)
            3: {"classes": 0, "confidence": 0.90}, # DIFFERS from pred_1_it (0 vs 1)
        }

        # We will simulate that the SelectionStrategy chose instances 0, 2, and 3
        self.selected_indices = np.array([0, 2, 3])

        # Pack kwargs exactly as the orchestrator would
        self.kwargs = {
            'predictions': self.predictions,
            'cl_memory': self.cl_memory,
            'pred_1_it': self.pred_1_it,
            'pred_x_it': self.pred_x_it
        }

    ###########################
    ##         BASE          ##
    ###########################
    def test_should_raise_type_error_when_try_to_call_label_instances_in_superclass(
        self,
    ):  # NOQA
        with self.assertRaises(TypeError) as err:
            LabelingStrategy()

    def test_should_raise_not_implemented_when_try_to_call_label_instances_in_superclass(
        self,
    ):  # NOQA
        with self.assertRaises(NotImplementedError) as err:
            StrategyNoImplementMock().label_instances(np.array([]))

    def test_should_initiate_when_try_to_call_label_instances(self):
        self.assertTrue(
            StrategyImplementMock().label_instances(np.array([]))
        )

    def test_naive_strategy(self):
        """
        Test that NaiveStrategy directly applies the current iteration's prediction.
        """
        strategy = NaiveStrategy()
        labels = strategy.label_instances(self.selected_indices, **self.kwargs)

        # It should just take the values from self.predictions at indices [0, 2, 3]
        # self.predictions = [2, 0, 1, 0] -> indices [0, 2, 3] -> [2, 1, 0]
        expected_labels = np.array([2, 1, 0])

        assert_array_equal(labels, expected_labels)

    def test_memory_strategy(self):
        """
        Test that MemoryStrategy pulls the most frequent class from historical memory.
        """
        strategy = MemoryStrategy()
        labels = strategy.label_instances(self.selected_indices, **self.kwargs)

        # It should take the argmax of cl_memory for indices [0, 2, 3]
        # cl_memory[0] -> argmax([1.0, 0.0, 5.0]) -> 2
        # cl_memory[2] -> argmax([0.0, 4.0, 0.0]) -> 1
        # cl_memory[3] -> argmax([2.0, 2.0, 6.0]) -> 2
        expected_labels = np.array([2, 1, 2])

        assert_array_equal(labels, expected_labels)

    def test_rule_based_strategy_mixed_conditions(self):
        """
        Test that RuleBasedLabelStrategy correctly toggles between Naive (when classes match)
        and Memory (when classes diverge) logic.
        """
        strategy = RuleBasedLabelStrategy()
        selected_indices = [2, 3]

        labels = strategy.label_instances(np.array(selected_indices), **self.kwargs)

        # Breakdown of expected behavior for selected_indices [0, 2, 3]:
        # Index 0: pred_1 (2) == pred_x (2). Matches! Use pred_1_it -> Class 2
        # Index 2: pred_1 (1) != pred_x (2). Differs! Use memory argmax -> Class 1
        # Index 3: pred_1 (1) != pred_x (0). Differs! Use memory argmax -> Class 2

        expected_labels = np.array([1, 2])

        assert_array_equal(labels, expected_labels)

    def test_rule_based_strategy_all_match(self):
        """
        Test RuleBasedLabelStrategy when all selected instances have matching predictions.
        (Simulating instances selected strictly by Rules 1 or 2).
        """
        strategy = RuleBasedLabelStrategy()

        # Force selection of indices 0 and 1 (both match in our setUp data)
        selected_match_indices = np.array([0, 1])
        labels = strategy.label_instances(selected_match_indices, **self.kwargs)

        # Index 0 matches -> pred_1_it class -> 2
        # Index 1 matches -> pred_1_it class -> 0
        expected_labels = np.array([2, 0])

        assert_array_equal(labels, expected_labels)

    def test_rule_based_strategy_all_differ(self):
        """
        Test RuleBasedLabelStrategy when all selected instances have differing predictions.
        (Simulating instances selected strictly by Rules 3 or 4).
        """
        strategy = RuleBasedLabelStrategy()

        # Force selection of indices 2 and 3 (both differ in our setUp data)
        selected_differ_indices = np.array([2, 3])
        labels = strategy.label_instances(selected_differ_indices, **self.kwargs)

        # Index 2 differs -> cl_memory argmax -> 1
        # Index 3 differs -> cl_memory argmax -> 2
        expected_labels = np.array([1, 2])

        assert_array_equal(labels, expected_labels)
