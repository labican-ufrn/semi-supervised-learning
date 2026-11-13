import numpy as np

from mlabican.selection.base import SelectionStrategy


class Rules(SelectionStrategy):
    """
    FlexConC specific logic: Evaluates predictions based on iteration
        memory and 4 strict rules, which are:
    - Rule 1: Classes equal AND both confidences > thr
    - Rule 2: Classes equal AND at least one > thr (excluding
        instances already in Rule 1)
    - Rule 3: Classes differ AND both > thr
    - Rule 4: Classes differ AND at least one > thr (excluding
        instances already in Rule 3)

    Each rule is only executed if, and only if, the previous one does
        not select any instance. If no instance is selected, an empty
        list was returned.
    """

    def __init__(self) -> None:
        super().__init__()
        self.insertion_rules = [
            self._rule_1,
            self._rule_2,
            self._rule_3,
            self._rule_4,
        ]

    def select_instances(
        self, probabilities: np.ndarray, threshold: float = 0.95, **kwargs
    ) -> np.ndarray:
        probs_1_it = kwargs.get('prob_1_it', [])
        max_proba_1_it = np.max(probs_1_it, axis=1) >= threshold
        labels_1_it = np.argmax(probs_1_it, axis=1)

        max_proba_x_it = np.max(probabilities, axis=1) >= threshold
        labels_x_it = np.argmax(probabilities, axis=1)

        for rule in self.insertion_rules:
            selected = rule(
                labels_1_it, labels_x_it, max_proba_1_it, max_proba_x_it
            )
            if any(selected):
                if self.verbose:
                    msg = (
                        f'Instances 1st > thr: {np.sum(max_proba_1_it)}'
                        f'Instances Xst > thr: {np.sum(max_proba_x_it)}'
                        f'Selected: {len(selected)}'
                    )
                    self.logger.info(msg)
                return np.where(selected)[0]

        return np.array([])

    def _rule_1(self, lbls_1_it, lbls_x_it, thr_1_it, thr_x_it) -> list[int]:
        equal_labels = lbls_1_it == lbls_x_it
        both_high = thr_1_it & thr_x_it

        return equal_labels & both_high

    def _rule_2(self, lbls_1_it, lbls_x_it, thr_1_it, thr_x_it) -> list[int]:
        equal_labels = lbls_1_it == lbls_x_it
        at_least_one_high = thr_1_it | thr_x_it

        return equal_labels & at_least_one_high

    def _rule_3(self, lbls_1_it, lbls_x_it, thr_1_it, thr_x_it) -> list[int]:
        differ_labels = lbls_1_it != lbls_x_it
        both_high = thr_1_it & thr_x_it

        return differ_labels & both_high

    def _rule_4(self, lbls_1_it, lbls_x_it, thr_1_it, thr_x_it) -> list[int]:
        differ_labels = lbls_1_it != lbls_x_it
        at_least_one_high = thr_1_it | thr_x_it

        return differ_labels & at_least_one_high
