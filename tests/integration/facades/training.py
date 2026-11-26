from sklearn.gaussian_process import GaussianProcessClassifier as Naive

from mlabican.flexcon import FlexCon
from tests.integration.facades.base import BaseFacade
from tests.integration.helpers.metrics import compute_metric


class TrainingFacade(BaseFacade):
    """Facade for training a model in the context of BDD tests."""
    def __init__(self, context) -> None:
        super().__init__(context)
        self.model = None

    def __algorithm_chooser(self, algorithm_name: str, **kwargs):
        """Choose the algorithm based on the provided algorithm name."""
        algorithm_name = algorithm_name.lower()

        if algorithm_name == "flexcon-c":
            return FlexCon(**kwargs)

        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    def __model_chooser(self, model_name: str):
        """Choose the model based on the provided model name."""
        model_name = model_name.lower()

        if model_name == "naive":
            return Naive()

        raise ValueError(f"Unknown model: {model_name}")

    def initialize(self, algorithm: str, model: str) -> None:
        """Create an algorithm instance with the specified model."""
        self.model = self.__algorithm_chooser(
            algorithm_name=algorithm,
            estimator=self.__model_chooser(model),
            verbose=False,
        )

    def train(self) -> None:
        try:
            self.model.fit(
                self.context.dataset.data,
                self.context.dataset.target,
            )
        except Exception as exc:
            self.context.last_error = exc

    def evaluate(self, metrics: list[str]) -> list[float]:
        y_pred = self.context.training_facade.model.predict(
            self.context.dataset.data
        )
        y_prob = self.context.training_facade.model.predict_proba(
            self.context.dataset.data
        )

        return [
            compute_metric(
                metric,
                self.context.dataset_y_true,
                y_pred,
                y_prob,
            )
            for metric in metrics
        ]
