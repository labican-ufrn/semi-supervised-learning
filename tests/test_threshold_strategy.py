from unittest import TestCase

from mlabican.threshold.base import ThresholdStrategy

from mlabican.threshold.classifier import Classifier
from mlabican.threshold.flexcon import FlexConRatio
from mlabican.threshold.gradual import Gradual


class StrategyNoImplementMock(ThresholdStrategy):
    def update_threshold(self, current_threshold: float, cr: float = 0.05, **kwargs) -> float:
        return super().update_threshold(current_threshold, cr)


class StrategyImplementMock(ThresholdStrategy):
    def update_threshold(self, current_threshold: float, cr: float = 0.05, **kwargs) -> float:
        return True


class TestThresholdStrategy(TestCase):
    ###########################
    ##         BASE          ##
    ###########################
    def test_should_raise_type_error_when_try_create_threshold_object_without_implement_abstract_methods(
        self,
    ):  # NOQA
        with self.assertRaises(TypeError) as err:
            ThresholdStrategy()

    def test_should_raise_not_implemented_when_try_to_call_update_threshold_in_superclass(
        self,
    ):  # NOQA
        with self.assertRaises(NotImplementedError) as err:
            StrategyNoImplementMock().update_threshold(.95, .5)

    def test_should_initiate_when_try_to_call_select_instances(self):
        self.assertTrue(StrategyImplementMock().update_threshold(.95, .5))

    ###########################
    ##        GRADUAL        ##
    ###########################
    def test_should_decrease_threshold_by_cr(self):
        strategy = Gradual()
        cr = 0.05
        old_thr = 0.95

        new_thr = strategy.update_threshold(old_thr, cr)

        self.assertEqual(new_thr, old_thr - cr)
        self.assertGreater(old_thr, new_thr)

    def test_should_zero_threshold_when_decrease_pass_zero(self):
        strategy = Gradual()
        cr = 0.3
        old_thr = 0.04

        new_thr = strategy.update_threshold(old_thr, cr)

        self.assertEqual(new_thr, 0.0)
        self.assertGreater(old_thr, new_thr)

    def test_gradual_should_not_change_threshold_when_cr_is_zero(self):
        strategy = Gradual()
        cr = 0.0
        old_thr = 0.9

        new_thr = strategy.update_threshold(old_thr, cr)

        self.assertEqual(new_thr, old_thr)

    ###########################
    ##      CLASSIFIER       ##
    ###########################
    def test_classifier_should_not_change_threshold_when_cr_is_zero(self):
        strategy = Classifier()
        cr = 0.0
        old_thr = 0.9
        kwargs = {
            'local_measure': 0.9,
            'init_measure': 0.95,
        }

        new_thr = strategy.update_threshold(old_thr, cr, **kwargs)

        self.assertEqual(new_thr, old_thr)

    def test_classifier_should_decrease_threshold_when_local_measure_greater_than_init(self):
        strategy = Classifier()
        cr = 0.05
        old_thr = 0.9
        kwargs = {
            'local_measure': 0.98,
            'init_measure': 0.95,
        }

        new_thr = strategy.update_threshold(old_thr, cr, **kwargs)

        self.assertEqual(new_thr, old_thr - cr)
        self.assertGreater(old_thr, new_thr)

    def test_classifier_should_decrease_threshold_when_local_measure_greater_than_init_limit_lower_bound(self):
        strategy = Classifier()
        cr = 0.5
        old_thr = 0.4
        kwargs = {
            'local_measure': 0.98,
            'init_measure': 0.95,
        }

        new_thr = strategy.update_threshold(old_thr, cr, **kwargs)

        self.assertEqual(new_thr, 0.0)
        self.assertGreater(old_thr, new_thr)

    def test_classifier_should_increase_threshold_when_local_measure_lower_than_init(self):
        strategy = Classifier()
        cr = 0.05
        old_thr = 0.9
        kwargs = {
            'local_measure': 0.8,
            'init_measure': 0.95,
        }

        new_thr = strategy.update_threshold(old_thr, cr, **kwargs)

        self.assertEqual(new_thr, old_thr + cr)
        self.assertLess(old_thr, new_thr)

    def test_classifier_should_increase_threshold_when_local_measure_lower_than_init_limit_upper_bound(self):
        strategy = Classifier()
        cr = 0.5
        old_thr = 0.9
        kwargs = {
            'local_measure': 0.8,
            'init_measure': 0.95,
        }

        new_thr = strategy.update_threshold(old_thr, cr, **kwargs)

        self.assertEqual(new_thr, 1.0)
        self.assertLess(old_thr, new_thr)

    def test_classifier_should_not_change_threshold_when_local_measure_near_to_init(self):
        strategy = Classifier()
        cr = 0.05
        old_thr = 0.9

        # Equal case
        kwargs = {
            'local_measure': 0.95,
            'init_measure': 0.95,
        }
        new_thr = strategy.update_threshold(old_thr, cr, **kwargs)

        self.assertEqual(new_thr, old_thr)

        # Lower bound
        kwargs = {
            'local_measure': 0.94,
            'init_measure': 0.95,
        }
        new_thr = strategy.update_threshold(old_thr, cr, **kwargs)

        self.assertEqual(new_thr, old_thr)

        # Upper bound
        kwargs = {
            'local_measure': 0.96,
            'init_measure': 0.95,
        }
        new_thr = strategy.update_threshold(old_thr, cr, **kwargs)

        self.assertEqual(new_thr, old_thr)

    ###########################
    ##        FLEXCON        ##
    ###########################
    def test_flexcon_strategy_should_update_thr(self):
        strategy = FlexConRatio()

        current_threshold = 0.9

        new_thr = strategy.update_threshold(
            current_threshold,
            coverage=0.5,
            avg_predict_proba=0.8,
        )
        self.assertAlmostEqual(new_thr, 0.7333333333333334) # (0.9 + 0.5 + 0.8) / 3

    def test_flexcon_should_stay_old_thr_when_average_prob_is_zero(self):
        strategy = FlexConRatio()

        current_threshold = 0.9

        new_thr = strategy.update_threshold(
            current_threshold,
            coverage=0.5,
            avg_predict_proba=0.0,
        )
        self.assertEqual(new_thr, 0.90)

    def test_flexcon_should_keep_the_thr_when_all_parameters_are_equal(self):
        strategy = FlexConRatio()

        current_threshold = 1.0

        new_thr = strategy.update_threshold(
            current_threshold,
            coverage=1.0,
            avg_predict_proba=1.0,
        )
        self.assertAlmostEqual(new_thr, 1.0)

        current_threshold = .5

        new_thr = strategy.update_threshold(
            current_threshold,
            coverage=.5,
            avg_predict_proba=.5,
        )
        self.assertAlmostEqual(new_thr, .5)
