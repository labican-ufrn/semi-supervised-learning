# MLabican

**MLabican** is the Machine Learning library used at Labican. It is developed in Python and designed to facilitate the use of semi-supervised learning algorithms — a highly effective approach when you have a massive amount of data, but only a small portion of it is labeled.

The library implements a family of algorithms based on Self-Training. In this approach, the model itself can label unknown data based on its confidence. MLabican also provides advanced versions featuring pseudo-label revaluation, quality filters, and classifier ensembles (committees) for more robust decision-making.

[TOC]
- [When to Use MLabican?](#when-to-use-mlabican)
- [Installation](#installation)
- [Available Algorithms](#available-algorithms)
- [Important Parameters](#important-parameters)
- [Tips & Best Practices](#tips--best-practices)
- [Development & Contributing](#development--contributing)
  - [Prerequisites](#prerequisites)
  - [Virtual Environment Setup](#virtual-environment-setup)
  - [Install Dependencies](#install-dependencies)
  - [Pre-commit Hooks](#pre-commit-hooks)
- [Licence](#licence)

---

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

| Python Class / Interface | Category | Description |
|:-------------------------|:---------|:------------|
| FlexCon / SelfFlexCon    | Core     | A flexible self-training implementation that delegates instance selection, labeling, thresholding, and health monitoring to injected strategies. |
| Ensemble                 | Core     | Wrapper class containing multiple FlexCon classifiers to manage ensemble weighting and predictions. |
| LabelingStrategy         | Strategy | Interface defining how new instances receive labels (Implementations: `MemoryStrategy`, `NaiveStrategy`, `RuleBasedLabelStrategy`). |
| SelectionStrategy        | Strategy | Determines which pseudo-labeled instances should be kept. The Rules implementation specifically handles complex selection by applying multi-rule filtering logic for NumPy arrays based on specific probability thresholds. |
| ThresholdStrategy        | Strategy | Interface for dynamically updating the confidence threshold over iterations (Implementations: `Classifier`, `FlexConRatio`, `Gradual`). |
| ModelHealthMonitor       | Strategy | Monitors model health across iterations to prevent degradation, tracking metric shifts (Implementations: `PredictionDriftMonitor`, `StabilityMonitor`). |
| RevaluationStrategy      | Strategy | Reassesses the quality of pseudo-labels, utilizing specific metrics to drop or keep instances (Implementations: `Immediate`, `Deferred`). |
| DifficultMetrics         | Utility  | Defines mathematical calculations to score instance difficulty or cluster quality (Implementations: `Silhouette`, `DaviesBouldin`). |

### Example
1. Preparing the Data

MLabican algorithms work with a partially labeled target array (y). To indicate unlabeled data, you must use -1 for your unknown labels.

```python
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
    cr=0.05,
    # Dependency Injection for Strategies
    # if all none, it's FlexCon-C default version
    threshold_strategy=None,  # Classifier()
    selection_strategy=None,  # Rules()
    labeling_strategy=None,   # RuleBasedLabelStrategy()
    # Other parameters
    threshold=0.95,
    max_iter=100,
    verbose=True  # Generate a log file with result for each iteration
)


flexcon.fit(iris.data, iris.target)

print('Finish test')

```


## Important Parameters
| Parameter | Description |
|:----------|:------------|
| threshold | Minimum probability required to consider a pseudo-label as confident. |
| cr        | change rate, it changes the threshold based on this value. |
| max_iter  | Maximum number of iterations the algorithm will run. |
| verbose   | If True, prints logs detailing the iterations and decisions. |
| others    | Check each Strategy classes to make custom configurations. |

## Tips & Best Practices
- Always use `-1` to mark unknown labels.
- The `estimator` passed to the models must implement the `.predict_proba()` method.
- Use `verbose=True` while experimenting to better understand the algorithm's behavior.

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

Once finished, when you try to commit, the pre-commit hooks will analyze your files and will fix some minor problems to improve and standardize the code based on the patterns defined in the `.pyproject` file.

## Licence

This library is licensed under the **MIT License**, permitting free use, including for commercial purposes.
