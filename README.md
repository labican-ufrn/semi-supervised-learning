# 📚 Guia do Usuário — Biblioteca `mlabican`

## 🔍 O que é `mlabican`?

A `mlabican` é uma biblioteca Python desenvolvida para facilitar o uso de **algoritmos de aprendizado semissupervisionado** — uma abordagem útil quando você possui muitos dados, mas poucos estão rotulados.

A biblioteca implementa uma família de algoritmos baseados em **Self-Training (auto-treinamento)**, onde o próprio modelo é capaz de **rotular dados desconhecidos** com base em sua confiança, além de versões avançadas com **reavaliação de pseudo-rótulos**, **filtros de qualidade** e **comitês de classificadores** para decisões mais robustas.

## ✅ Quando usar?

Você deve considerar o uso da `mlabican` quando:

- Possui muitos dados, mas a rotulação é cara ou limitada.
- Quer aproveitar melhor o conjunto de dados parcialmente rotulado.
- Precisa de modelos mais robustos, que filtrem e corrijam rótulos imprecisos.

## 💾 Instalação

A instalação é feita via [PyPI](https://pypi.org/project/mlabican/) com `pip`. Execute no terminal:

```bash
pip install mlabican
```

## 🧠 Principais Algoritmos disponíveis

A biblioteca oferece quatro classes principais de algoritmos:

| Classe Python                            | O que faz                                                                 |
|------------------------------------------|---------------------------------------------------------------------------|
| `SelfTrainingClassifier`                 | Auto-treinamento clássico: adiciona rótulos confiáveis à medida que aprende. |
| `SelfWithRevaluation`                    | Auto-treinamento + reavaliação de pseudo-rótulos usando índice de silhueta. |
| `SelfWithRevaluationEssemble`            | Adiciona um comitê de classificadores que votam em instâncias ruidosas para serem reavaliadas.     |
| `SelfWithRevaluationEssembleWeights`     | Mesmo funcionamento do SelfWithRevaluationEssemble porém os classificadores possuem pesos de votação proporcionais a sua acurácia inicial do dataset.             |


## 🛠️ Como preparar os dados

Os algoritmos da `mlabican` funcionam com `y` parcialmente rotulado. Para indicar **dados não rotulados**, use `np.nan` ou `-1` nos rótulos.

```python
import numpy as np
y_train = y_train.astype(float)
rng = np.random.default_rng(42)
y_train[rng.choice(len(y_train), size=30, replace=False)] = np.nan  # ou = -1
```

## 🚀 Exemplo rápido: SelfTrainingClassifier

```python
from mlabican.selfTraining import SelfTrainingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Dados
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Tornando parte dos dados não rotulados
y_train = y_train.astype(float)
y_train[:20] = np.nan

# Modelo de base
base_model = DecisionTreeClassifier()

# Self-training
clf = SelfTrainingClassifier(base_estimator=base_model, threshold=0.9)
clf.fit(X_train, y_train)

# Predição
y_pred = clf.predict(X_test)
```

## 🔄 Versões com Reavaliação e Comitê

Você pode usar modelos mais sofisticados com reclassificação por comitê:

```python
from mlabican.selfTraining import SelfWithRevaluationEssemble
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Comitê de classificadores
committee = VotingClassifier([
    ("dt", DecisionTreeClassifier()),
    ("nb", GaussianNB()),
    ("svc", SVC(probability=True))
])

clf = SelfWithRevaluationEssemble(
    base_estimator=DecisionTreeClassifier(),
    committee=committee,
    threshold=0.9,
    max_iter=10,
    verbose=True
)
clf.fit(X_train, y_train)
```

## ⚙️ Parâmetros importantes

| Parâmetro         | O que faz                                                                 |
|-------------------|--------------------------------------------------------------------------|
| `threshold`       | Probabilidade mínima para considerar um rótulo como confiável.           |
| `k_best`          | Número fixo de exemplos mais confiáveis a rotular (caso use `criterion="k_best"`). |
| `criterion`       | `"threshold"` ou `"k_best"`. Define a forma de seleção dos exemplos.     |
| `max_iter`        | Número máximo de iterações.                                               |
| `silhouette_threshold` | Define a qualidade mínima dos pseudo-rótulos (entre 0 e 1) (usado nas versões com reavaliação). |
| `verbose`         | Se `True`, exibe logs das iterações e decisões.                          |

## ✅ Dicas

- Prefira usar `np.nan` ou `-1` para marcar rótulos desconhecidos.
- O `base_estimator` deve implementar `.predict_proba()`.
- Use `verbose=True` para entender o comportamento do algoritmo.
- A versão com comitê pode melhorar resultados quando os dados têm ruído.


## 🧪 Testes

A lib já acompanha um arquivo de testes (`testSelf.py`) cobrindo todos os algoritmos. Você pode rodá-lo para verificar se está tudo funcionando corretamente.

## 📄 Licença

Esta biblioteca é licenciada sob a **Licença MIT**, permitindo uso livre, inclusive para fins comerciais.
