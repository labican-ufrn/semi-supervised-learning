# MLabican

**MLabican** is the Machine Learning library used at Labican. It is developed in Python and designed to facilitate the use of semi-supervised learning algorithms — an highly effective approach when you have a massive amount of data, but only a small portion of it is labeled.

The library implements a family of algorithms based on Self-Training. In this approach, the model itself can label unknown data based on its confidence. MLabican also provides advanced versions featuring pseudo-label revaluation, quality filters, and classifier ensembles (committees) for more robust decision-making.

## When to Use MLabican?

You should consider using mlabican when:

- You have a large dataset, but labeling is expensive, time-consuming, or limited.

- You want to maximize the potential of a partially labeled dataset.

- You need robust models capable of filtering out and correcting inaccurate pseudo-labels.

## Installation

You can easily install the library via PyPI using pip:
```sh
pip install mlabican
```

## Available Algorithms
The library offers four main classes of algorithms:

TODO: Update this table

| Python Class | Description |
|:-------------|:------------|
|SelfTrainingClassifier|Classic Self-Training: Iteratively adds confident labels to the training set as it learns.|
|SelfWithRevaluation|Self-Training + Revaluation: Re-evaluates pseudo-labels using the silhouette index to ensure quality.|
|SelfWithRevaluationEssemble|Ensemble Revaluation: Adds a committee of classifiers that vote on noisy instances to determine if they should be re-evaluated.|
|SelfWithRevaluationEssembleWeights|Weighted Ensemble Revaluation: Similar to the standard Ensemble, but the voting classifiers have weights proportional to their initial accuracy on the dataset.|

### Example
1. Preparing the Data

MLabican algorithms work with a partially labeled target array (`y`). To indicate unlabeled data, you must use `np.nan` or `-1` (preferred) in your labels.

```python
import numpy as np

y_train = y_train.astype(float)
rng = np.random.default_rng(42)

# Mask 30 random instances as unlabeled data
y_train[rng.choice(len(y_train), size=30, replace=False)] = -1  # or = np.nan
```

2. Quick Start: SelfTrainingClassifier

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from mlabican.selfTraining import SelfTrainingClassifier

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Make a portion of the data unlabeled
y_train = y_train.astype(float)
y_train[:20] = np.nan

# Define the base model
base_model = DecisionTreeClassifier()

# Initialize and fit the Self-training classifier
clf = SelfTrainingClassifier(estimator=base_model, threshold=0.9)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
```

3. Advanced Version: Revaluation & Ensemble

You can use more sophisticated models that feature reclassification by a committee. This is particularly useful when your data contains noise.

```python
from mlabican.selfTraining import SelfWithRevaluationEssemble
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Define a committee of classifiers
committee = VotingClassifier([
    ("dt", DecisionTreeClassifier()),
    ("nb", GaussianNB()),
    ("svc", SVC(probability=True))
])

# Initialize the advanced self-training model
clf = SelfWithRevaluationEssemble(
    estimator=DecisionTreeClassifier(),
    committee=committee,
    threshold=0.9,
    max_iter=10,
    verbose=True
)

clf.fit(X_train, y_train)
```


## Important Parameters
| Parameter | Description |
|:----------|:------------|
| threshold | Minimum probability required to consider a pseudo-label as confident. |
| k_best | A fixed number of the most confident examples to label per iteration (used if criterion="k_best"). |
| criterion | "threshold" or "k_best". Defines how examples are selected for labeling. |
| max_iter | Maximum number of iterations the algorithm will run. |
| silhouette_threshold | Defines the minimum quality of the pseudo-labels (between 0 and 1). Used in versions with revaluation. |
| verbose | If True, prints logs detailing the iterations and decisions. |

## Tips & Best Practices
- Always use `np.nan` or `-1` (preferable) to mark unknown labels.
- The `estimator` passed to the models must implement the `.predict_proba()` method.
- Use `verbose=True` while experimenting to better understand the algorithm's behavior.
- The Ensemble (Committee) versions generally yield better results when dealing with noisy datasets.

## Development & Contributing

If you want to contribute to MLabican, please follow these guidelines:

### Prerequisites

We use Python 3.12 for development. The library may not work correctly on previous versions of Python.

### Virtual Environment Setup
We highly recommend using Python Virtual Environments. While you can use your favorite tool (`pyenv`, `poetry`, etc.), we provide instructions for `venv` as it is built-in and beginner-friendly:

```sh
python -m venv .venv
# Activate it (Linux/macOS/WSL)
source .venv/bin/activate
```

```sh
python -m venv .venv
# Activate it (Windows)
.venv\Scripts\activate
```

### Install Dependencies

Once activated, install the required development packages:
```sh
pip install -e ".[dev]"
```

### Pre-commit Hooks

After installing the requirements, you need to configure the `pre-commit` hooks. This tool prevents bad or breaking commits from entering the repository and automatically fixes minor issues to standardize the code based on the patterns defined in our `.pyproject` file.

```sh
pre-commit install --install-hooks
```

Finish, when you try to commit now the pre-commit hooks will analyse your files
and will fix some minor problems to improve and standardize the code based on the
patters defined in the `.pyproject` file.

## Licence

This library is licensed under the **MIT License**, permitting free use, including for commercial purposes.
