from mlabican.selfTraining import SelfTrainingClassifier
from mlabican.selfNew import SelfWithRevaluation
from mlabican.selfNewEssemble import SelfWithRevaluationEssemble
from mlabican.selfNewEssembleCP import SelfWithRevaluationEssembleWeights

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def make_unlabeled(y, percent=0.3, seed=42):
    y_mod = y.astype(float)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(y_mod), size=int(percent * len(y_mod)), replace=False)
    y_mod[indices] = -1
    return y_mod


def test_classifier(name, model_cls, use_committee=False):
    print(f"\n====== Testando {name} ======")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_train = make_unlabeled(y_train, 0.3)

    base = DecisionTreeClassifier(random_state=42)

    if use_committee:
        committee = VotingClassifier([
            ("dt", DecisionTreeClassifier()),
            ("nb", GaussianNB()),
            ("svc", SVC(probability=True))
        ])
        model = model_cls(base_estimator=base, committee=committee)
    else:
        model = model_cls(base_estimator=base)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.2f}")
    print(f"Condição de parada: {model.termination_condition_}")


if __name__ == "__main__":
    test_classifier("SelfTrainingClassifier", SelfTrainingClassifier)
    test_classifier("SelfWithRevaluation", SelfWithRevaluation)
    test_classifier("SelfWithRevaluationEssemble", SelfWithRevaluationEssemble, use_committee=True)
    test_classifier("SelfWithRevaluationEssembleWeights", SelfWithRevaluationEssembleWeights, use_committee=True)
