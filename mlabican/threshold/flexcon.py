from mlabican.threshold.base import ThresholdStrategy


class FlexConRatio(ThresholdStrategy):
    """
    FlexConRatio was firstly introduced in the FlexCon algorithm. The
    threshold is updated based on a weighted average of:
    - current_threshold: the threshold.
    - avg_predict_proba: the quality of the last batch of predictions
        (average confidence);
    - coverage: ratio between labeled in the current iteration and
        remain unlabeled data
    """

    def update_threshold(
        self, current_threshold: float, cr: float = 0.05, **kwargs
    ) -> float:
        coverage = kwargs.get('coverage', 1.0)
        avg_predict_proba = kwargs.get('avg_predict_proba', 1.0)

        if avg_predict_proba == 0:
            return current_threshold

        new_thr = (current_threshold + coverage + avg_predict_proba) / 3

        return min(1.0, max(0.0, new_thr))
