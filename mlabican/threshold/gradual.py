from mlabican.threshold.base import ThresholdStrategy


class Gradual(ThresholdStrategy):
    """
    Gradual Strategy was firstly introduced by FlexConG where the
    threshold decreases the threshold by `cr` factor each iteration.
    """

    def update_threshold(
        self, current_threshold: float, cr: float = 0.05, **kwargs
    ) -> float:
        return max(0.0, current_threshold - cr)
