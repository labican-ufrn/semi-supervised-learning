import numpy as np
from behave.runner import Context

from tests.integration.facades.dataset import DatasetFacade
from tests.integration.facades.training import TrainingFacade


class MlabicanContext(Context):
    """Custom typed context for Behave BDD tests."""
    dataset_facade: DatasetFacade
    training_facade: TrainingFacade

    metrics: dict[str, float]

    dataset_y_true: np.ndarray
    predictions: np.ndarray
