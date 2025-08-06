from random import choice

from numpy import where
from pandas import read_csv
from selfNew import MySelfNew
from selfNewEssemble import MySelfNewEssemble
from selfNewEssembleCP import MySelfNewEssembleCP
from selfOld import MySelfOld
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils import select_labels

# Separa/prepara a base de dados
df = read_csv('datasets/Iris.csv', header=0)
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# Inicializa os classificadores
classifiers = {
    "KNN": KNeighborsClassifier,
    "Naive Bayes": GaussianNB,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "XGBoost": XGBClassifier,
    "Logistic Regression": LogisticRegression,
    "Neural Network": MLPClassifier,
}


# Dividir o dataset (5% rotulados, 95% não rotulados)SelfTrainingClassifier
# Alterei para um split 85-15 para dados de treinamento e teste.
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
)


# Todas as instâncias em X_train estão rotuladas. Isso é injusto para
# nível de comparação dos métodos. Para corrigir isso, eu vou fazer um
# novo split nos dados deixando apenas 5% dos 85% (mesmo que o
# select_labels). A partir deste novo split é que eu faço o treinamento
# e teste do meu algoritmo/comitê.
#
# Nos dados de treinamento eu vou escolher 5% para "ficar com rótulo"
X_train, _, y_train, _ = train_test_split(
    X_train_all,
    y_train_all,
    test_size=0.95,
    random_state=42,
    stratify=y_train_all
)


print(f"\ntamanho de X_train: {len(X_train)}")
print(f"tamanho de y_train: {len(y_train)}\n")


# Treina os classificadores com o percentual de instâncias inicialmente rotuladas
results = {name: [] for name in classifiers.keys()}

for name, model in classifiers.items():

    if name == "Logistic Regression":
        model = model(max_iter=1000)
    elif name == "Neural Network":
        model = model(max_iter=3000)
    else:
        model = model()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test_all)
    acc = accuracy_score(y_test_all, y_pred)
    results[name].append(acc)
    print(f"{name}: {results[name]}")

# Seleciona o melhor classificador
max_acc = max(results.values())
best_models = [name for name, acc in results.items() if acc == max_acc]
best_model_name = choice(best_models)
best_model = classifiers[best_model_name]


##################################################


# Cria o comitê
models = []
for name, model in classifiers.items():

    if name == "Logistic Regression":
        model = model(max_iter=1000)
    elif name == "Neural Network":
        model = model(max_iter=3000)
    else:
        model = model()

    models.append((name, model))

# Treina o comitê com o percentual inicial de instâncias rotuladas
# committee = VotingClassifier(estimators=models, voting='soft', verbose=False)
# committee.fit(X_train, y_train)
# Avalia acurácia de cada modelo no conjunto rotulado
weights = []
for name, model in models:
    model_clone = clone(model)
    model_clone.fit(X_train, y_train)
    y_pred = model_clone.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    weights.append(acc)

# Normaliza os pesos
total_weight = sum(weights)
normalized_weights = [w / total_weight for w in weights]

# Cria comitê ponderado
committee = VotingClassifier(
    estimators=models,
    voting='soft',
    weights=normalized_weights,
    verbose=False
)

committee.fit(X_train, y_train)


##################################################


# Inicia e treina o especialista com o percentual inicial de instâncias rotuladas
specialist = MySelfNewEssembleCP(
    base_estimator=best_model(),
    committee=committee,
    threshold=0.95,
    max_iter=100,
    silhouette_threshold=-0.2,
    verbose=True
)

y_2 = select_labels(y_train_all, X_train_all, 0.05)

# Contar instâncias não rotuladas antes do treinamento
num_unlabeled_before = sum(where(y_2 == -1, 1, 0))
print(f"\nInstâncias não rotuladas antes do treinamento: {num_unlabeled_before}\n")

specialist.fit(X_train_all, y_2)

# Contar instâncias rotuladas e não rotuladas após o treinamento
y_pseudo = specialist.transduction_
num_unlabeled_after = sum(where(y_pseudo == -1, 1, 0))
print(f"\nInstâncias não rotuladas após o treinamento: {num_unlabeled_after}")
print(f"Quantidade de instâncias que foram rotuladas após o treinamento: {num_unlabeled_before - num_unlabeled_after}")

print(
    f'\nTotal de instâncias: {X.shape}'
    f'\nTotal de selecionadas para o experimento: {X_train_all.shape}'
    f'\nTotal de rotuladas que foram para o comitê: {X_train.shape}'
    f'\nTotal de rotuladas que foram para o especialista: {sum(where(y_2 != -1, 1, 0))}'
)

# Armazena as predições do especialista
y_pred = specialist.predict(X_test_all)

# Calcula a acurácia do especialista
accuracy = accuracy_score(y_test_all, y_pred)
print(f"\nAcurácia do Especialista ({best_model_name}): {accuracy:.4f}")

print(f"Critério de parada: {specialist.termination_condition_}")
print(f"Número de iterações realizadas: {specialist.n_iter_}")
