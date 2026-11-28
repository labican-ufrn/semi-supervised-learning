import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

CONFIGS = {
    'accuracy': (accuracy_score, True, {}),
    'auc': (roc_auc_score, False, {'multi_class': 'ovr'}),
    'f1-score': (f1_score, True, {'average': 'macro'}),
}

def compute_metric(
    metric_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Compute the specified metric using the provided true labels, predicted labels, and predicted probabilities.

    Args:
        metric_name (str): The name of the metric to compute.
        y_true (np.ndarray): The true labels of the dataset.
        y_pred (np.ndarray): The predicted labels of the dataset.
        y_prob (np.ndarray): The predicted probabilities of the dataset.

    Raises:
        ValueError: If the provided metric name is unknown.

    Returns:
        float: The computed metric value rounded to 4 decimal places.
    """
    if metric_name not in CONFIGS:
        raise ValueError(f"Unknown metric: {metric_name}")

    method, is_pred, kwargs = CONFIGS[metric_name]
    target_y = y_pred if is_pred else y_prob

    return round(method(y_true, target_y, **kwargs), 4)
