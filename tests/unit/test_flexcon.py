from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from mlabican.flexcon import FlexCon


def array_side_effect(items):
    it = iter(items)
    return lambda *args, **kwargs: next(it)


class ClassifierMock(MagicMock):
    def predict(self, X):
        return np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

    # def predict_proba(self, X):
    #     return np.array([
    #         [0.3, 0.7],
    #         [0.2, 0.8],
    #         [0.1, 0.9],
    #         [0.4, 0.6],
    #         [0.5, 0.5],
    #         [0.6, 0.4],
    #         [0.7, 0.3],
    #         [0.8, 0.2],
    #         [0.9, 0.1],
    #         [0.99, 0.01],
    #     ])


class GenerateMemory():
    def pred_1_it(self):
        return {
            1: {
                'confidence': 0.3,
                'classes': 0
            },
            2: {
                'confidence': 0.2,
                'classes': 0
            },
            3: {
                'confidence': 0.97,
                'classes': 0
            },
            4: {
                'confidence': 0.97,
                'classes': 1
            },
            5: {
                'confidence': 0.98,
                'classes': 1
            }
        }

    def pred_x_it(self):
        return {
            1: {
                'confidence': 0.3,
                'classes': 0
            },
            2: {
                'confidence': 0.92,
                'classes': 1
            },
            3: {
                'confidence': 0.96,
                'classes': 1
            },
            4: {
                'confidence': 0.7,
                'classes': 1
            },
            5: {
                'confidence': 0.99,
                'classes': 1
            }
        }


class TestFlexCon(TestCase):
    @patch("mlabican.flexcon.clone")
    @patch("mlabican.flexcon.SelfTrainingClassifier")
    def setUp(self, super_class, model_clone):
        super_class.return_value = ClassifierMock()
        model_clone.return_value = ClassifierMock()

        selection_strategy = MagicMock()
        labeling_strategy = MagicMock()
        threshold_strategy = MagicMock()

        self.flexcon = FlexCon(
            estimator=ClassifierMock(),
            threshold_strategy=threshold_strategy,
            selection_strategy=selection_strategy,
            labeling_strategy=labeling_strategy,
            max_iter=10,
            cr=0.05,
            threshold=1.0,
            verbose=False
        )

        self.flexcon.transduction_ = np.full(10, -1)
        self.flexcon.labeled_iter_ = np.full(10, -1)

    def test_should_return_updated_cl_memory_by_weights_when_weights_are_passed(self):  # NOQA
        instances = list(range(10))
        labels = [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
        weights = [0.3, 0.2, 0.5, 0.6, 0.7, 0.1, 0.8, 0.4, 0.9, 0.57]
        expected_output_weights = [
            [0.3, 0],
            [0.2, 0],
            [0.5, 0],
            [0, 0.6],
            [0, 0.7],
            [0, 0.1],
            [0, 0.8],
            [0, 0.4],
            [0.9, 0],
            [0, 0.57],
        ]
        self.flexcon.cl_memory = [[0] * 2 for _ in range(len(instances))]
        self.flexcon.update_memory(instances, labels, weights)
        self.assertListEqual(self.flexcon.cl_memory, expected_output_weights)

    def test_should_return_updated_cl_memory_by_one_when_no_weights_are_passed(self):  # NOQA
        instances = list(range(10))
        labels = [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
        labels2 = [1, 1, 0, 1, 0, 1, 0, 0, 1, 1]
        output_without_weights = [
            [1, 1],
            [1, 1],
            [2, 0],
            [0, 2],
            [1, 1],
            [0, 2],
            [1, 1],
            [1, 1],
            [1, 1],
            [0, 2],
        ]
        self.flexcon.cl_memory = [[0] * 2 for _ in range(len(instances))]
        self.flexcon.update_memory(instances, labels)
        self.flexcon.update_memory(instances, labels2)
        self.assertListEqual(self.flexcon.cl_memory, output_without_weights)

    def test_storage_predict_should_return_valid_struct(self,):
        expected_output = {
            1: {
                'confidence': 0.952,
                'classes': 0
            },
            2: {
                'confidence': 0.45,
                'classes': 1
            },
        }

        instances = [1, 2]
        confidences = [0.952, 0.45]
        classes = [0, 1]

        self.assertDictEqual(
            self.flexcon.storage_predict(instances, confidences, classes),
            expected_output
        )

    def test_should_return_local_measure_when_call_local_measure(self):
        instances = list(range(10))
        labels = [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
        expected_local_measure = 0.2

        self.assertAlmostEqual(
            self.flexcon.calc_local_measure(instances, labels),
            expected_local_measure
        )

    def test_should_update_labels_for_existing_instances(self):
        selected_instances = list(range(3,7))
        pseudo_labels = [0, 1, 1, 0]

        self.flexcon.add_new_labeled(selected_instances, pseudo_labels)
        self.assertListEqual(
            self.flexcon.transduction_.tolist(),
            [-1, -1, -1, 0, 1, 1, 0, -1, -1, -1]
        )
        self.assertListEqual(
            self.flexcon.labeled_iter_.tolist(),
            [-1, -1, -1, 0, 0, 0, 0, -1, -1, -1]
        )

    @patch("mlabican.flexcon.FlexCon.calc_local_measure")
    def test_fit_reduces_threshold_when_zero_selected(self, local_measure):
        local_measure.return_value = 0.9
        self.flexcon.max_iter = 2
        self.flexcon.update_memory = MagicMock()

        self.flexcon.selection_strategy.select_instances.side_effect = [
            np.array([], dtype=int),  # Iter 1: 0 selected
            np.array([0, 1]),  # Iter 2: both selected
        ]
        self.flexcon.labeling_strategy.label_instances.return_value = np.array(
            [0, 1]
        )
        self.flexcon.threshold_strategy.update_threshold.return_value = 0.5

        self.flexcon.estimator_.predict_proba.return_value = np.array([
            [0.3, 0.7],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.99, 0.01],
        ])

        self.flexcon.fit(np.random.rand(10, 2), self.flexcon.transduction_)

        self.assertLess(self.flexcon.threshold, 1.0)
        # self.assertEqual(self.flexcon.threshold, 0.99)
        self.assertEqual(self.flexcon.update_memory.call_count, 1)

    @patch("mlabican.flexcon.FlexCon.calc_local_measure")
    def test_fit_stops_at_max_iter(self, local_measure):
        local_measure.return_value = 0.9
        self.flexcon.max_iter = 2
        self.flexcon.update_memory = MagicMock()

        self.flexcon.estimator_.predict_proba.return_value = np.array([
            [0.3, 0.7],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.99, 0.01],
        ])

        self.flexcon.selection_strategy.select_instances.return_value = np.array(
            [], dtype=int
        )

        self.flexcon.fit(np.random.rand(10, 2), self.flexcon.transduction_)

        self.assertEqual(self.flexcon.n_iter_, 2)
        self.assertEqual(self.flexcon.termination_condition_, 'Max iterations')
        # Unlabeled instances remain -1
        self.assertListEqual(
            self.flexcon.transduction_.tolist(),
            [-1 for _ in range(10)]
        )

    @patch("mlabican.flexcon.FlexCon.calc_local_measure")
    def test_fit_labels_all_instances_successfully(self, local_measure):
        self.X = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ])
        self.y = np.array([-1, -1, -1, -1, -1, -1])

        self.flexcon.max_iter = 10
        self.flexcon.threshold = 0.9

        self.flexcon.estimator_.predict_proba.side_effect = [
            np.array([
                [0.9, 0.1],  # class 0 - selected
                [0.4, 0.6],  # class 1
                [0.7, 0.3],  # class 0
                [0.8, 0.2],  # class 0
                [0.2, 0.8],  # class 1
                [0.1, 0.9],  # class 1 - selected
            ]),  # Iter 1
            np.array([
                [0.4, 0.6],  # class 1
                [0.7, 0.3],  # class 0
                [0.8, 0.2],  # class 0 - selected
                [0.2, 0.8],  # class 1 - selected
            ]),  # Iter 2

            np.array([
                [0.6, 0.4],  # class 0
                [0.7, 0.3],  # class 0
            ]),  # Iter 3
        ]
        # self.flexcon.estimator_.predict.side_effect = [
        #     np.array([0, 1]),  # Iter 1
        #     np.array([1]),  # Iter 2
        # ]

        local_measure.return_value = 0.9
        self.flexcon.selection_strategy.select_instances.side_effect = [
            np.array([0, 5]),
            np.array([2, 3]),
            np.array([0, 1]),
        ]
        self.flexcon.labeling_strategy.label_instances.side_effect = [
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0, 0]),
        ]
        self.flexcon.threshold_strategy.update_threshold.side_effect = [0.8, 0.7, 0.5]
        self.flexcon.update_memory = MagicMock()


        result = self.flexcon.fit(self.X, self.y)

        # Assertions
        self.assertIs(result, self.flexcon)
        self.assertEqual(self.flexcon.n_iter_, 3)
        self.assertEqual(
            self.flexcon.termination_condition_, 'Label all instances'
        )
        self.assertListEqual(
            self.flexcon.transduction_.tolist(),
            [0, 0, 0, 0, 1, 1]
        )
        self.assertListEqual(
            self.flexcon.labeled_iter_.tolist(),
            [1, 3, 3, 2, 2, 1]
        )
        self.assertEqual(self.flexcon.threshold, 0.5)
        # Estimator should fit twice in the loop + 1 final fit at the end = 3
        self.assertEqual(self.flexcon.estimator_.fit.call_count, 4)
