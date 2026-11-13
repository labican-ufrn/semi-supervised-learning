from mlabican.threshold.base import ThresholdStrategy


class Classifier(ThresholdStrategy):
    """
    Classifier Strategy was firstly introduced by FlexConC where the
    threshold is changed based on the local measure vs initial measure.
    Suppose that the measure is accuracy, for example.

    Defines:
    - local acc: calculated by classifier trained in the current
    iteration in the initial labeled instances.
    - initial acc: calculated by classifier trained and evaluated with
    the initial labeled instances.

    When the local acc is higher than initial acc it is possible to
    decrease the threshold, because the method assume that the knowledge
    is enough (the classifier made good generalization). On the other
    hand, when the local acc is lower than initial acc the algorithm
    will assume that the classifier does not made good generalization
    and the threshold must be increase to select most confidente
    instances and reduce "noise data" in the labeled instances.
    """

    def update_threshold(
        self, current_threshold: float, cr: float = 0.05, **kwargs
    ) -> float:
        local_measure: float = kwargs.get('local_measure', 0.0)
        init_measure: float = kwargs.get('init_measure', 0.0)

        if local_measure > init_measure + 0.01:
            return max(current_threshold - cr, 0.0)

        if local_measure < init_measure - 0.01:
            return min(current_threshold + cr, 1.0)

        return current_threshold
