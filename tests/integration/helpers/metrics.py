from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


def compute_metric(metric_name, y_true, y_pred, y_prob):
    configs = {
        'accuracy': (accuracy_score, True, {}),
        'auc': (roc_auc_score, False, {'multi_class': 'ovr'}),
        'f1-score': (f1_score, True, {'average': 'macro'}),
    }

    if metric_name not in configs:
        raise ValueError(f"Unknown metric: {metric_name}")

    method, is_pred, kwargs = configs[metric_name]
    target_y = y_pred if is_pred else y_prob

    return round(method(y_true, target_y, **kwargs), 4)
