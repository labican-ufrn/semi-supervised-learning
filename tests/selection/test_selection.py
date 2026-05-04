from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from mlabican.selection.selection import SelectionStrategy


class StrategyNoImplementMock(SelectionStrategy, MagicMock):
    def select_instances(self, probabilities, **kwargs):
        super().select_instances(probabilities)


class StrategyImplementMock(SelectionStrategy, MagicMock):
    def select_instances(self, probabilities, **kwargs):
        return True


class TestSelectionStrategy(TestCase):
    def test_should_raise_type_error_when_try_to_call_select_instances_in_superclass(self):  # NOQA
        with self.assertRaises(TypeError) as err:
            SelectionStrategy()

    def test_should_raise_not_implemented_when_try_to_call_select_instances_in_superclass(self):  # NOQA
        with self.assertRaises(NotImplementedError) as err:
            StrategyNoImplementMock().select_instances(np.array([]))

    def test_should_initiate_when_try_to_call_select_instances(self):
        self.assertTrue(
            StrategyImplementMock().select_instances(np.array([]))
        )
