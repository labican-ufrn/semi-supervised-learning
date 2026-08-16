import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier as Naive

from mlabican.flexcon import FlexCon
rng = np.random.RandomState(42)
iris = datasets.load_iris()
random_unlabeled_points = rng.rand(iris.target.shape[0]) < 0.3
iris.target[random_unlabeled_points] = -1

flexcon = FlexCon(
    estimator=Naive(),
    verbose=True
)

flexcon.fit(iris.data, iris.target)

print('Finish test')
