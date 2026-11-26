import numpy as np
from sklearn import datasets

from tests.integration.facades.base import BaseFacade


class DatasetFacade(BaseFacade):
    """Facade for dataset in the context of BDD tests."""

    def initialize(self, dataset: str, ssl_rate: int) -> None:
        """Initialize the dataset with the specified SSL rate."""
        rng = np.random.RandomState(42)
        iris = datasets.load_iris()
        y_true = iris.target.copy()
        print(f'Val {ssl_rate} - {type(ssl_rate)}')

        random_unlabeled_points = rng.rand(iris.target.shape[0]) < ssl_rate / 100
        iris.target[random_unlabeled_points] = -1

        self.context.dataset = iris
        self.context.dataset_y_true = y_true
