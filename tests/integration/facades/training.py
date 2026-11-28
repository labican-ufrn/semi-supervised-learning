from sklearn.gaussian_process import GaussianProcessClassifier as Naive

from mlabican.flexcon import FlexCon
from tests.integration.facades.base import BaseFacade
from tests.integration.helpers.metrics import compute_metric


class TrainingFacade(BaseFacade):
    """Facade for training a model in the context of BDD tests."""
    def __init__(self, context) -> None:
        super().__init__(context)
        self.model = None

    def __algorithm_chooser(
        self,
        algorithm_name: str,
        **kwargs,
    ) -> FlexCon:
        """Choose the algorithm based on the provided algorithm name.

        Args:
            algorithm_name (str): The name of the algorithm to choose.
                Currently, only 'flexcon-c' is supported.

        Raises:
            ValueError: If the provided algorithm name is unknown.

        Returns:
            FlexCon: The chosen algorithm instance.
        """
        algorithm_name = algorithm_name.lower()

        if algorithm_name == "flexcon-c":
            return FlexCon(**kwargs)

        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    def __model_chooser(self, model_name: str) -> Naive:
        """Choose the model based on the provided model name.

        Args:
            model_name (str): The name of the model to choose.
                Currently, only 'naive' is supported.

        Raises:
            ValueError: If the provided model name is unknown.

        Returns:
            Naive: The chosen model instance.
        """
        model_name = model_name.lower()

        if model_name == "naive":
            return Naive()

        raise ValueError(f"Unknown model: {model_name}")

    def initialize(self, algorithm: str, model: str) -> None:
        """Create an algorithm instance with the specified model.

        Args:
            algorithm (str): The name of the algorithm to initialize.
            model (str): The name of the model to use with the algorithm.
        """
        self.model = self.__algorithm_chooser(
            algorithm_name=algorithm,
            estimator=self.__model_chooser(model),
            verbose=False,
        )

    def train(self) -> None:
        """Train the model using the dataset in the context."""
        try:
            self.model.fit(
                self.context.dataset.data,
                self.context.dataset.target,
            )
        except Exception as exc:
            self.context.last_error = exc

    def evaluate(self, metrics: list[str]) -> list[float]:
        """Evaluate all metrics on the trained model using the dataset
        in the context.

        Args:
            metrics (list[str]): A metric list contain all metrics that
                should be calculated.

        Returns:
            list[float]: A list of metric values.
        """
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
