import numpy as np
from behave import given, then, when
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier as Naive
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from mlabican.flexcon import FlexCon


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



@given('que o algoritmo "{algorithm}" foi instanciado com o classificador "{model}"')
def create_algorithm(context, algorithm, model):
    context.flexcon = FlexCon(
        estimator=Naive(),
        verbose=False
    )


@given('que a base de dados "{dataset}" possuía {ssl_rate:n} por cento dos dados não rotulados')
def create_ssl_dataset(context, dataset, ssl_rate):
    rng = np.random.RandomState(42)
    iris = datasets.load_iris()
    y_true = iris.target.copy()
    print(f'Val {ssl_rate} - {type(ssl_rate)}')

    random_unlabeled_points = rng.rand(iris.target.shape[0]) < ssl_rate / 100
    iris.target[random_unlabeled_points] = -1

    context.dataset = iris
    context.dataset_y_true = y_true


@when('o algoritmo treinar')
def model_fit(context):
    context.flexcon.fit(
        context.dataset.data,
        context.dataset.target,
    )


@then('a performance da métrica deverá ser igual a:')
def evaluate(context):
    y_pred = context.flexcon.predict(context.dataset.data)
    y_prob = context.flexcon.predict_proba(context.dataset.data)

    for row in context.table.rows:
        score = compute_metric(
            row['metric'],
            context.dataset_y_true,
            y_pred,
            y_prob,
        )
        assert score == float(row['value']), (
            f'A métrica {row["metric"]} foi {score}, deveria ser {row["value"]}'
        )
