from random import choice

from numpy import where
from pandas import read_csv
from selfNewEssembleCP import MySelfNewEssembleCP
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils import select_labels

##################################################
# Configurações iniciais e preparação dos dados


# Separa/prepara a base de dados
df = read_csv('datasets/Iris.csv', header=0)
# separa os atributos (X) e o rótulo (y)
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


# Dividir o dataset em um split 85-15 para dados de treinamento e teste.
# Para fazer uma validação justa após todo o experimento, foi separado 15% dos
# dados. O especialista e o comitê serão treinados usando o restante dos dados
# rotulados.
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


##################################################
# Avalia qual dos classificadores será o especialista. Para o conjunto de dados.


# Cada classificador é treinado individualmente usando os dados rotulados. O
# processo de treinamento é realizado para cada classificador, e os resultados
# são armazenados para comparação posterior. O classificador com a melhor
# acurácia no conjunto de teste será selecionado como o especialista para o
# processo iterativo de rotulagem.
results: dict[str, list[float]] = {
    name: []
    for name in classifiers.keys()
}

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
# Cria e treina o comitê de classificadores.

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

# Treina o comitê com as instâncias selecionadas no X_train.
# Avalia acurácia de cada modelo no conjunto rotulado e calcula os pesos para o
# comitê com base na acurácia.
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
# Inicio do Self-Training com o especialista e o comitê.

# Inicia e treina o especialista com o percentual inicial de instâncias
# rotuladas
specialist = MySelfNewEssembleCP(
    base_estimator=best_model(),
    committee=committee,
    threshold=0.95,  # Valor inicial do threshold, o threshold é atualizado durante a execução do algoritmo.
    max_iter=100,
    silhouette_threshold=-0.2,
    verbose=True
)

# Seleciona 5% das instâncias para manter o rótulo
y_2 = select_labels(y_train_all, X_train_all, 0.05)

# Contar instâncias não rotuladas antes do treinamento
num_unlabeled_before = sum(where(y_2 == -1, 1, 0))
print(f"\nInstâncias não rotuladas antes do treinamento: {num_unlabeled_before}\n")

specialist.fit(X_train_all, y_2)

# Contar instâncias não rotuladas após o treinamento
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
