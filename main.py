import numpy as np
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier as Naive
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from mlabican.flexcon import FlexCon

rng = np.random.RandomState(42)
iris = datasets.load_iris()
y_true = iris.target.copy()

# percentual de instâncias que serão não-rotuladas
random_unlabeled_points = rng.rand(iris.target.shape[0]) < 0.2

iris.target[random_unlabeled_points] = -1

flexcon = FlexCon(
    estimator=Naive(),
    verbose=False
)

flexcon.fit(iris.data, iris.target)

y_pred = flexcon.predict(iris.data)
y_prob = flexcon.predict_proba(iris.data)

# Métricas que deverão ser reportadas
print(f' Acc: {accuracy_score(y_true, y_pred)}')
print(f'  F1: {f1_score(y_true, y_pred, average="macro")}')
print(f' Auc: {roc_auc_score(y_true, y_prob, multi_class="ovr")}')

print('Finish test')
